import pandas as pd
import requests
import json
from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
import pickle
import os
import base64
import google.generativeai as genai
import logging
import time
from google.api_core import exceptions, retry
from dotenv import load_dotenv
from email.mime.text import MIMEText
from html.parser import HTMLParser
import schedule
import boto3
import sqlite3
import re
from docx import Document
from bs4 import BeautifulSoup
import pyarrow.parquet as pq
import uuid

# Load environment variables
load_dotenv()

# Configuration from .env
SNOW_URL = os.getenv("SNOW_URL")
SNOW_USER = os.getenv("SNOW_USER")
SNOW_PASS = os.getenv("SNOW_PASS")
GMAIL_ADDRESS = os.getenv("GMAIL_ADDRESS")
MANAGER_EMAIL = os.getenv("MANAGER_EMAIL")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME")
S3_MODIFIED_BUCKET_NAME = os.getenv("S3_MODIFIED_BUCKET_NAME", "my-modified-bucket-2025")
PROCESSING_BATCH_SIZE = int(os.getenv("PROCESSING_BATCH_SIZE", 1000))
PROCESSING_INTERVAL = int(os.getenv("PROCESSING_INTERVAL", 5))

# Validate environment variables
required_vars = ["SNOW_URL", "SNOW_USER", "SNOW_PASS", "GMAIL_ADDRESS", "MANAGER_EMAIL", "GEMINI_API_KEY",
                 "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "S3_BUCKET_NAME"]
missing_vars = [var for var in required_vars if not os.getenv(var)]
if missing_vars:
    raise ValueError(f"Missing environment variables: {', '.join(missing_vars)}")

# Check S3 buckets
s3_client = boto3.client('s3', aws_access_key_id=AWS_ACCESS_KEY_ID, aws_secret_access_key=AWS_SECRET_ACCESS_KEY)
for bucket in [S3_BUCKET_NAME, S3_MODIFIED_BUCKET_NAME]:
    try:
        s3_client.head_bucket(Bucket=bucket)
    except:
        raise ValueError(f"S3 bucket {bucket} does not exist or is inaccessible")

# Set base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
LOG_FILE = os.path.join(BASE_DIR, 'logs', 'combined_log.txt')
TICKET_IDS_FILE = os.path.join(BASE_DIR, 'ticket_ids.txt')
WAREHOUSE_DB = os.path.join(BASE_DIR, 'warehouse.db')
QUERIED_DATA_FILE = os.path.join(DATA_DIR, 'queried_transactions.csv')
CHANGE_LOG_FILE = os.path.join(BASE_DIR, 'logs', 'change_log.txt')
SOLUTIONS_FILE = os.path.join(BASE_DIR, 'logs', 'solutions.json')

# Logging setup
logging.basicConfig(filename=LOG_FILE, level=logging.INFO, format='%(asctime)s: %(message)s')
change_logger = logging.getLogger('change_logger')
change_handler = logging.FileHandler(CHANGE_LOG_FILE)
change_handler.setFormatter(logging.Formatter('%(asctime)s: %(message)s'))
change_logger.addHandler(change_handler)
change_logger.setLevel(logging.INFO)


def log_action(message):
    logging.info(message)
    print(message)


def log_change(message):
    change_logger.info(message)


# Gmail API setup
SCOPES = ['https://www.googleapis.com/auth/gmail.modify', 'https://www.googleapis.com/auth/gmail.send']


def get_gmail_service():
    creds = None
    token_path = os.path.join(BASE_DIR, 'token.pickle')
    creds_path = os.path.join(BASE_DIR, 'credentials.json')

    try:
        log_action("Attempting to set up Gmail service...")
        if os.path.exists(token_path):
            with open(token_path, 'rb') as token:
                creds = pickle.load(token)
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                log_action("Refreshing expired credentials...")
                creds.refresh(Request())
            else:
                if not os.path.exists(creds_path):
                    log_action(f"Missing {creds_path}. Download from Google Cloud Console.")
                    return None
                log_action("Initiating OAuth flow...")
                flow = InstalledAppFlow.from_client_secrets_file(creds_path, SCOPES)
                creds = flow.run_local_server(port=0)
            with open(token_path, 'wb') as token:
                pickle.dump(creds, token)
        service = build('gmail', 'v1', credentials=creds)
        log_action("Gmail service initialized successfully.")
        return service
    except Exception as e:
        log_action(f"Error setting up Gmail service: {e}")
        return None


# Gemini API setup
try:
    log_action("Setting up Gemini API...")
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel('gemini-1.5-flash')
    log_action("Gemini API initialized successfully.")
except Exception as e:
    log_action(f"Error setting up Gemini API: {e}")
    model = None


# Retry decorator for Gemini API
@retry.Retry(predicate=retry.if_exception_type(exceptions.ResourceExhausted), initial=46, maximum=120, multiplier=2)
def call_gemini_with_retry(prompt):
    return model.generate_content(prompt)


# Define workflows for Capital One data issues
WORKFLOWS = {
    "missing_values": {
        "detect": lambda df: df[['transaction_id', 'amount', 'transaction_date']].isnull().any().any(),
        "solution": [{"step": "Fill missing amounts with median", "action": "fill_median_amount"}],
        "priority": "high",
        "resolvable": True,
        "team": "Data Quality Team"
    },
    "incorrect_date_format": {
        "detect": lambda df: not pd.to_datetime(df['transaction_date'], errors='coerce').notnull().all(),
        "solution": [{"step": "Convert transaction_date to datetime", "action": "fix_date"}],
        "priority": "normal",
        "resolvable": True,
        "team": "Data Quality Team"
    },
    "duplicate_transactions": {
        "detect": lambda df: df.duplicated(subset=['transaction_id']).any(),
        "solution": [{"step": "Remove duplicate transactions", "action": "drop_duplicates"}],
        "priority": "high",
        "resolvable": True,
        "team": "Data Quality Team"
    },
    "outlier_amounts": {
        "detect": lambda df: (df['amount'] > df['amount'].quantile(0.99) * 2).any(),
        "solution": [{"step": "Flag outlier amounts for review", "action": "flag_outliers"}],
        "priority": "high",
        "resolvable": True,
        "team": "Fraud Detection Team"
    },
    "compliance_violation": {
        "detect": lambda df: df['transaction_type'].isnull().any(),
        "solution": [{"step": "Set default transaction_type to Unknown", "action": "set_default_type"}],
        "priority": "critical",
        "resolvable": True,
        "team": "Data Quality Team"
    }
}


def create_ticket(table, subject, description, priority, team):
    try:
        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json"
        }
        assignment_group = "data_quality_team_id" if team == "Data Quality Team" else "fraud_detection_team_id"
        data = {
            "short_description": subject,
            "description": description,
            "assignment_group": assignment_group,
            "urgency": "1" if priority in ["high", "critical"] else "2",
            "impact": "1" if priority in ["high", "critical"] else "2",
            "state": "1"
        }
        url = f"{SNOW_URL}/{table}"
        log_action(f"Creating {table} ticket: {subject}, assigning to {team}")
        response = requests.post(url, auth=(SNOW_USER, SNOW_PASS), headers=headers, json=data)
        if response.status_code == 201:
            result = response.json()['result']
            log_action(f"Ticket Created: {result['number']} (ID: {result['sys_id']}) for {team}")
            return result['sys_id'], result['number']
        else:
            log_action(f"Error creating {table} ticket: {response.status_code} - {response.text}")
            return None, None
    except Exception as e:
        log_action(f"Error creating ticket: {e}")
        return None, None


def update_ticket(table, ticket_number, status, comment, priority='normal'):
    try:
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        state_map = {
            "New": "1", "In Progress": "2", "On Hold": "3",
            "Resolved": "6", "Closed": "7", "Cancelled": "8"
        }
        data = {
            "state": state_map.get(status, "1"),
            "comments": comment,
            "priority": "1" if priority in ["high", "critical"] else "4"
        }
        url = f"{SNOW_URL}/{table}/{ticket_number}"
        log_action(f"Updating {table} ticket {ticket_number} to {status}")
        response = requests.patch(url, auth=(SNOW_USER, SNOW_PASS), headers=headers, json=data)
        if response.status_code == 200:
            log_action(f"Updated {table} ticket {ticket_number} to {status}")
        else:
            log_action(f"Error updating {table} ticket {ticket_number}: {response.status_code} - {response.text}")
    except Exception as e:
        log_action(f"Error updating ticket: {e}")


def send_approval_email(service, ticket_number, subject, description):
    try:
        log_action(f"Preparing approval email for ticket {ticket_number} to {MANAGER_EMAIL}")
        message = MIMEText(
            f"Please review this {('change request' if ticket_number.startswith('CHG') else 'incident')}:\n"
            f"Title: {subject}\n"
            f"Description: {description}\n\n"
            f"Reply 'Approved' or 'Denied' to this email."
        )
        message['From'] = GMAIL_ADDRESS
        message['To'] = MANAGER_EMAIL
        message['Subject'] = f"Approval Needed for Ticket {ticket_number}"
        raw_message = base64.urlsafe_b64encode(message.as_bytes()).decode()
        log_action(f"Sending approval email for ticket {ticket_number}")
        service.users().messages().send(
            userId='me',
            body={'raw': raw_message}
        ).execute()
        log_action(f"Sent approval email for ticket {ticket_number} to {MANAGER_EMAIL}")
    except Exception as e:
        log_action(f"Error sending approval email: {e}")


def analyze_data(df):
    issues = []
    for issue, workflow in WORKFLOWS.items():
        if workflow["detect"](df):
            issues.append({
                "issue": issue,
                "solution": workflow["solution"],
                "priority": workflow["priority"],
                "resolvable": workflow["resolvable"],
                "team": workflow["team"]
            })
    if not issues and model:
        prompt = f"""
        Analyze this financial dataset summary: {df.to_string()}
        Identify data issues relevant to a bank like Capital One and suggest solutions in JSON format:
        - issue: Describe the issue (e.g., "potential_fraud")
        - solution: Array of steps (e.g., [{"step": "Flag for fraud review", "action": "flag_fraud"}])
        - priority: "high", "normal", or "critical"
        - resolvable: Boolean (true if AI can resolve, false if human intervention needed)
        - team: "Data Quality Team" or "Fraud Detection Team"
        """
        try:
            log_action("Analyzing data with Gemini...")
            response = call_gemini_with_retry(prompt)
            text = response.text.strip()
            if text.startswith('```json'):
                text = text[7:].rstrip('```').strip()
            result = json.loads(text)
            issues.append(result)
            log_action(f"Gemini analysis result: {result}")
        except Exception as e:
            log_action(f"Error analyzing data with Gemini: {e}")
    return issues


def resolve_issue(df, issue):
    original_df = df.copy()
    solution = issue["solution"]
    log_action(f"Attempting to resolve issue: {issue['issue']}")
    modified_indices = []

    for step in solution:
        action = step.get("action")
        if action == "fill_median_amount":
            if pd.to_numeric(df['amount'], errors='coerce').isna().any():
                log_action("Invalid amount data detected, skipping fill_median_amount")
                return df, False
            median_amount = df['amount'].median()
            modified_indices.extend(df[df['amount'].isnull()].index.tolist())
            df['amount'].fillna(median_amount, inplace=True)
        elif action == "fix_date":
            modified_indices.extend(df[pd.to_datetime(df['transaction_date'], errors='coerce').isnull()].index.tolist())
            df['transaction_date'] = pd.to_datetime(df['transaction_date'], errors='coerce')
        elif action == "drop_duplicates":
            duplicates = df.duplicated(subset=['transaction_id'])
            modified_indices.extend(df[duplicates].index.tolist())
            df.drop_duplicates(subset=['transaction_id'], inplace=True)
        elif action == "flag_outliers":
            if pd.to_numeric(df['amount'], errors='coerce').isna().any():
                log_action("Invalid amount data detected, skipping flag_outliers")
                return df, False
            outliers = df['amount'] > df['amount'].quantile(0.99) * 2
            modified_indices.extend(df[outliers].index.tolist())
            df['outlier_flag'] = outliers
        elif action == "set_default_type":
            missing_type = df['transaction_type'].isnull()
            modified_indices.extend(df[missing_type].index.tolist())
            df['transaction_type'].fillna('Unknown', inplace=True)
        elif action == "flag_fraud":
            if pd.to_numeric(df['amount'], errors='coerce').isna().any():
                log_action("Invalid amount data detected, skipping flag_fraud")
                return df, False
            fraud = df['amount'] > 10000
            modified_indices.extend(df[fraud].index.tolist())
            df['fraud_flag'] = fraud
        else:
            log_action(f"Unknown action {action}, skipping")
            return df, False

    # Log changes
    if modified_indices:
        df['modified'] = False
        df.loc[modified_indices, 'modified'] = True
        log_change(f"File: {os.getenv('LATEST_FILE_KEY', 'unknown')}, Issue: {issue['issue']}")
        log_change(f"Original data for modified rows:\n{original_df.loc[modified_indices].to_string()}")
        log_change(f"Modified data for rows:\n{df.loc[modified_indices].to_string()}")
        log_change(f"Solution applied: {json.dumps(solution, indent=2)}")

    return df, True


def mask_sensitive_data(df):
    sensitive_columns = ['ssn', 'account_number']
    for col in sensitive_columns:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: '****' if pd.notnull(x) else x)
    log_action(
        "Masked sensitive data in columns: " + ", ".join([col for col in sensitive_columns if col in df.columns]))
    return df


def load_to_warehouse(df):
    try:
        conn = sqlite3.connect(WAREHOUSE_DB)
        df.to_sql('transactions', conn, if_exists='replace', index=False)
        conn.close()
        log_action(f"Data loaded to warehouse: {WAREHOUSE_DB}")
    except Exception as e:
        log_action(f"Error loading to warehouse: {e}")


def execute_sql_query(query):
    try:
        conn = sqlite3.connect(WAREHOUSE_DB)
        df = pd.read_sql_query(query, conn)
        conn.close()
        log_action(f"Executed SQL query: {query}\nResult:\n{df.to_string()}")
        return df
    except Exception as e:
        log_action(f"Error executing SQL query: {e}")
        return None


def save_queried_data(df, filename):
    try:
        df.to_csv(filename, index=False)
        log_action(f"Queried data saved to {filename}")
    except Exception as e:
        log_action(f"Error saving queried data: {e}")


def upload_to_modified_s3(df, original_file_key):
    try:
        file_extension = os.path.splitext(original_file_key)[1].lower()
        modified_file_key = f"processed_{os.path.basename(original_file_key)}"
        local_path = os.path.join(DATA_DIR, modified_file_key)

        if file_extension in ['.csv']:
            df.to_csv(local_path, index=False)
        elif file_extension in ['.json']:
            df.to_json(local_path, orient='records')
        elif file_extension in ['.parquet']:
            df.to_parquet(local_path)
        else:
            df.to_csv(local_path, index=False)  # Default to CSV for unsupported formats

        s3_client.upload_file(local_path, S3_MODIFIED_BUCKET_NAME, modified_file_key)
        log_action(f"Uploaded modified data to s3://{S3_MODIFIED_BUCKET_NAME}/{modified_file_key}")
        os.remove(local_path)
    except Exception as e:
        log_action(f"Error uploading to modified S3: {e}")


def read_file(file_path):
    file_extension = os.path.splitext(file_path)[1].lower()
    log_action(f"Reading file {file_path} with extension {file_extension}")

    try:
        if file_extension in ['.csv']:
            return pd.read_csv(file_path)
        elif file_extension in ['.json']:
            with open(file_path, 'r') as f:
                data = json.load(f)
                return pd.json_normalize(data)
        elif file_extension in ['.txt']:
            with open(file_path, 'r') as f:
                lines = [line.strip().split(',') for line in f if line.strip()]
                return pd.DataFrame(lines[1:], columns=lines[0] if lines else [])
        elif file_extension in ['.parquet']:
            return pq.read_table(file_path).to_pandas()
        elif file_extension in ['.html']:
            with open(file_path, 'r') as f:
                soup = BeautifulSoup(f, 'html.parser')
                table = soup.find('table')
                return pd.read_html(str(table))[0] if table else pd.DataFrame()
        elif file_extension in ['.docx']:
            doc = Document(file_path)
            data = [p.text.split(',') for p in doc.paragraphs if ',' in p.text]
            return pd.DataFrame(data[1:], columns=data[0] if data else [])
        else:
            raise ValueError(f"Unsupported file format: {file_extension}")
    except Exception as e:
        log_action(f"Error reading file {file_path}: {e}")
        return pd.DataFrame()


def get_latest_s3_files():
    try:
        response = s3_client.list_objects_v2(Bucket=S3_BUCKET_NAME)
        if 'Contents' not in response:
            log_action("No files found in S3 bucket")
            return []
        files = sorted(response['Contents'], key=lambda x: x['LastModified'], reverse=True)
        log_action(f"Found {len(files)} files in S3 bucket")
        return [f['Key'] for f in files]
    except Exception as e:
        log_action(f"Error checking S3: {e}")
        return []


def download_s3_file(file_key):
    local_path = os.path.join(DATA_DIR, file_key)
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    s3_client.download_file(S3_BUCKET_NAME, file_key, local_path)
    log_action(f"Downloaded {file_key} to {local_path}")
    return local_path


def process_data():
    service = get_gmail_service()
    if not service:
        log_action("Failed to initialize Gmail service, exiting.")
        return

    try:
        # Check S3 for latest files
        files = get_latest_s3_files()
        if not files:
            log_action("No new files to process")
            return

        for file_key in files[:PROCESSING_BATCH_SIZE]:  # Process batch of files
            os.environ['LATEST_FILE_KEY'] = file_key
            local_file = download_s3_file(file_key)
            if not os.path.exists(local_file):
                log_action(f"Failed to download {file_key}")
                continue

            # Read data based on file type
            df = read_file(local_file)
            if df.empty:
                log_action(f"Empty or invalid data from {file_key}, skipping")
                continue

            # Validate data types
            if 'amount' in df.columns and not pd.to_numeric(df['amount'], errors='coerce').notna().all():
                log_action(f"Invalid amount data in {file_key}, flagging for review")
                df['amount'] = pd.to_numeric(df['amount'], errors='coerce')
            if 'transaction_date' in df.columns:
                df['transaction_date'] = pd.to_datetime(df['transaction_date'], errors='coerce')

            log_action(f"Initial data from {file_key}:\n{df.head().to_string()}")

            # Initialize modified flag
            if 'modified' not in df.columns:
                df['modified'] = False
            if 'fraud_flag' not in df.columns:
                df['fraud_flag'] = False
            if 'outlier_flag' not in df.columns:
                df['outlier_flag'] = False

            # Mask sensitive data
            df = mask_sensitive_data(df)

            # Analyze for issues
            issues = analyze_data(df)
            unresolved_issues = []

            if not issues:
                log_action(f"No issues found in data from {file_key}")
                load_to_warehouse(df)
            else:
                # Load existing ticket IDs
                existing_tickets = set()
                if os.path.exists(TICKET_IDS_FILE):
                    with open(TICKET_IDS_FILE, 'r') as f:
                        for line in f:
                            line = line.strip()
                            if line and ' - ' in line:
                                ticket_number = line.split(' - ')[-1]
                                existing_tickets.add(ticket_number)

                for issue in issues:
                    issue_name = issue["issue"]
                    priority = issue["priority"]
                    resolvable = issue["resolvable"]
                    team = issue["team"]

                    # Attempt to resolve the issue
                    df, resolved = resolve_issue(df, issue)

                    if resolved:
                        log_action(f"Issue {issue_name} resolved automatically in {file_key}")
                    else:
                        unresolved_issues.append(issue)
                        log_action(f"Issue {issue_name} could not be resolved in {file_key}")

            # Load to data warehouse
            load_to_warehouse(df)
            log_action(f"Processed data from {file_key} loaded to warehouse")

            # Upload modified data to separate S3 bucket
            upload_to_modified_s3(df, file_key)

            # Handle unresolved issues
            if unresolved_issues:
                solutions = {}
                for issue in unresolved_issues:
                    issue_name = issue["issue"]
                    priority = issue["priority"]
                    team = issue["team"]
                    subject = f"Capital One Unresolved Data Issue: {issue_name} from {file_key}"
                    description = f"Unable to resolve issue in transaction data from {file_key}: {issue_name}\nData:\n{df.to_string()}"

                    if priority == "critical":
                        send_approval_email(service, "Pending", subject, description)
                        log_action(f"Critical issue {issue_name} reported to manager from {file_key}")
                    else:
                        ticket_id, ticket_number = create_ticket("incident", subject, description, priority, team)
                        if not ticket_number:
                            log_action(f"Failed to create ticket for {issue_name} from {file_key}")
                            continue
                        if ticket_number in existing_tickets:
                            log_action(f"Skipping duplicate ticket number: {ticket_number} for {file_key}")
                            continue

                        with open(TICKET_IDS_FILE, 'a') as f:
                            f.write(f"{subject} - {ticket_number}\n")

                        update_ticket("incident", ticket_number, "In Progress", f"Assigned to {team} for resolution",
                                      priority)

                        # Simulate manual resolution
                        if issue_name == "potential_fraud":
                            df['fraud_flag'] = df['amount'] > 10000
                            modified_indices = df[df['amount'] > 10000].index.tolist()
                            df.loc[modified_indices, 'modified'] = True
                            log_change(
                                f"File: {file_key}, Manual resolution for ticket {ticket_number}: Flagged potential fraud\nModified rows:\n{df.loc[modified_indices].to_string()}")
                        else:
                            unresolved_rows = df[df['modified'] == False]
                            log_change(
                                f"File: {file_key}, Unresolved rows for ticket {ticket_number}:\n{unresolved_rows.to_string()}")

                        update_ticket("incident", ticket_number, "Resolved",
                                      f"Resolved by {team}: {json.dumps(issue, indent=2)}", priority)

                    # Log possible solutions for unresolved issues
                    if not resolved:
                        issue_id = str(uuid.uuid4())
                        prompt = f"""
                        Provide possible manual steps to resolve the issue: {issue_name} in this dataset: {df.to_string()}
                        Return in JSON format:
                        - issue_id: {issue_id}
                        - issue: {issue_name}
                        - reason_unresolved: Why the AI couldn't resolve it
                        - steps: Array of manual steps (e.g., [{"step": "Review with team", "details": "Check customer ID"}])
                        - timestamp: Current timestamp in ISO format
                        - file_reference: {file_key}
                        """
                        try:
                            response = call_gemini_with_retry(prompt)
                            text = response.text.strip()
                            if text.startswith('```json'):
                                text = text[7:].rstrip('```').strip()
                            solution = json.loads(text)
                            solutions[issue_id] = solution
                            log_action(
                                f"Logged solution for {issue_name} (ID: {issue_id}) in solutions.json from {file_key}")
                        except Exception as e:
                            log_action(f"Error generating solution for {issue_name} from {file_key}: {e}")

                if solutions:
                    existing_solutions = {}
                    if os.path.exists(SOLUTIONS_FILE):
                        with open(SOLUTIONS_FILE, 'r') as f:
                            existing_solutions = json.load(f)
                    existing_solutions.update(solutions)
                    with open(SOLUTIONS_FILE, 'w') as f:
                        json.dump(existing_solutions, f, indent=2)

            # Execute SQL query and save results
            query = "SELECT * FROM transactions WHERE amount > 5000 OR fraud_flag = 1"
            queried_df = execute_sql_query(query)
            if queried_df is not None and not queried_df.empty:
                save_queried_data(queried_df, QUERIED_DATA_FILE)

    except Exception as e:
        log_action(f"Error processing data: {e}")


def main():
    log_action("Starting CombinedAgent for Capital One...")
    schedule.every(PROCESSING_INTERVAL).minutes.do(process_data)
    while True:
        schedule.run_pending()
        time.sleep(60)


if __name__ == "__main__":
    main()