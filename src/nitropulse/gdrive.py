import os
# Set this BEFORE importing anything else
os.environ['OAUTHLIB_INSECURE_TRANSPORT'] = '1'

# Import necessary libraries for Google Drive API
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

SCOPES = ['https://www.googleapis.com/auth/drive']

def get_service(path='./data/gdrive/'):
    creds = None

    token_path = os.path.join(path, 'token.json')
    creds_path = os.path.join(path, 'gdrive_credentials.json')
    
    # Check if we already have a token
    if os.path.exists(token_path):
        creds = Credentials.from_authorized_user_file(token_path, SCOPES)
    
    # If no valid credentials, authenticate
    if not creds or not creds.valid:
        flow = InstalledAppFlow.from_client_secrets_file(creds_path, SCOPES)
        
        # Set the redirect URI explicitly
        flow.redirect_uri = 'http://localhost'
        
        # Get the authorization URL
        auth_url, _ = flow.authorization_url(prompt='consent')
        print(f'Go to this URL: {auth_url}')
        print('After authorizing, you will be redirected to a localhost URL that won\'t load.')
        print('Copy the ENTIRE URL from your browser address bar and paste it here:')
        redirect_response = input('Full redirect URL: ')
        
        # Extract the authorization code from the redirect URL
        flow.fetch_token(authorization_response=redirect_response)
        creds = flow.credentials
        
        # Save for next time
        with open(token_path, 'w') as token:
            token.write(creds.to_json())
    
    return build('drive', 'v3', credentials=creds)


def create_folder(service, folder_name, parent_id=None):
    metadata = {
        'name': folder_name,
        'mimeType': 'application/vnd.google-apps.folder'
    }
    if parent_id:
        metadata['parents'] = [parent_id]
    
    folder = service.files().create(body=metadata, fields='id').execute()
    print(f"✅ Folder '{folder_name}' created with ID: {folder['id']}")
    return folder['id']


def make_folder_public(service, folder_id):
    permission = {
        'type': 'anyone',
        'role': 'reader'
    }
    service.permissions().create(
        fileId=folder_id,
        body=permission,
        fields='id'
    ).execute()
    print(f"🌍 Folder {folder_id} is now public.")


def get_shareable_link(service, folder_id):
    folder = service.files().get(fileId=folder_id, fields='webViewLink').execute()
    print("🔗 Public URL:", folder['webViewLink'])
    return folder['webViewLink']


if __name__ == '__main__':
    service = get_service()

    folder_name = "My Public Folder"
    folder_id = create_folder(service, folder_name)
    make_folder_public(service, folder_id)
    get_shareable_link(service, folder_id)