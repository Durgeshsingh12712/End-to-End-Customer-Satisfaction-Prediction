from pathlib import Path

CONFIG_FILE_PATH = Path("config/config.yaml")
PARAMS_FILE_PATH = Path("config/params.yaml")
SCHEMA_FILE_PATH = Path("schema.yaml")

# Column names
TICKET_ID = "Ticket ID"
CUSTOMER_NAME = "Customer Name"
CUSTOMER_EMAIL = "Customer Email"
CUSTOMER_AGE = "Customer Age"
CUSTOMER_GENDER = "Customer Gender"
PRODUCT_PURCHASED = "Product Purchased"
DATE_OF_PURCHASE = "Date of Purchase"
TICKET_TYPE = "Ticket Type"
TICKET_SUBJECT = "Ticket Subject"
TICKET_DESCRIPTION = "Ticket Description"
TICKET_STATUS = "Ticket Status"
RESOLUTION = "Resolution"
TICKET_PRIORITY = "Ticket Priority"
TICKET_CHANNEL = "Ticket Channel"
FIRST_RESPONSE_TIME = "First Response Time"
TIME_TO_RESOLUTION = "Time to Resolution"
CUSTOMER_SATISFACTION_RATING = "Customer Satisfaction Rating"

# Preprocessing constants
TARGET_COLUMN = "Ticket_Priority"
FEATURES_TO_DROP = ["Ticket ID", "Customer Name", "Customer Email", 
                   "Date of Purchase", "Resolution", "First Response Time", 
                   "Time to Resolution", "Customer Satisfaction Rating"]