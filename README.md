# ChatWithYourFiles

ChatWithYourFiles is a Streamlit-based web application that leverages OpenAI's GPT models and Chroma for document-based question answering. It allows users to interact with their documents in a conversational manner.

## Features

- **Document Processing**: Load and process various document formats.
- **AI-Powered Chat**: Utilize GPT models for answering questions based on the content of the loaded documents.
- **Chroma Integration**: Leverage Chroma for efficient document retrieval and indexing.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

- Python 3.x
- Pip (Python package installer)

### Installation

1. **Clone the Repository**
   ```bash
   git clone https://github.com/hugosmoreira/ChatWithYourFiles.git
   cd ChatWithYourFiles
Set Up a Virtual Environment (Optional but Recommended)

bash
Copy code
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
Install Required Packages

bash
Copy code
pip install -r requirements.txt
Environment Variables

Set up your .env file with the necessary API keys and configurations.
Running the Application
Start the Streamlit App

bash
Copy code
streamlit run app.py
Open Your Web Browser

Navigate to the URL provided in the terminal (usually http://localhost:8501).
Usage
Load Documents: Place your documents in the specified directory.
Ask Questions: Use the chat interface to ask questions about the contents of your documents.
Interact with AI: Receive answers generated by the AI based on your document's content.
Chroma Integration
Chroma is used for efficient indexing and retrieval of document content. It requires a setup for document processing and indexing.

Setting Up Chroma
Ensure you have Chroma installed and configured as per the project's requirements.
chroma.sqlite3 is used for data storage and indexing. This file should be generated during the first run of the application.
Excluding Chroma Database in Git
The chroma.sqlite3 file can be large and is excluded from the Git repository using .gitignore.
If you're setting up the project afresh, Chroma will create a new chroma.sqlite3 file.
Contributing
Contributions are welcome! For major changes, please open an issue first to discuss what you would like to change.

License
MIT


This README provides a basic structure and information for your project. Feel free to customize it to better fit your specific requirements, such as adding more detailed setup instructions, screenshots, or additional sections as needed.