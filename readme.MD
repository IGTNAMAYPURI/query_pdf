# Project Name: My Flask Application

## Description
This is a Python Flask application that utilizes machine learning techniques to perform operations on PDF files. The application allows users to upload PDF files and perform various tasks on them, such as finding matching text using semantic search and utilizing TensorFlow. The PDF data is stored in the form of chunks in JSON files, with each file named as "userId__chatId". The application provides several APIs to interact with the PDF processing functionalities.

## APIs
1. Upload PDF:
   - **Route**: `http://localhost:5000/upload?filename=userId__chatId`
   - **Method**: POST
   - **Form Data**: `file`
   - **Query**: `filename`
   - **Description**: This API allows the user to upload a PDF file for processing. The uploaded file should be sent as form data with the key `file`. The desired filename can be specified in the query parameter `filename`.

2. Generating Prompt:
   - **Route**: `http://localhost:5000/prompt?question=who is the author of the book&filename=userId__chatId`
   - **Method**: GET
   - **Query**: `?question=who is the author of the book&filename=userId__chatId`
   - **Description**: This API generates a prompt using the provided question and the specified PDF file. The question should be included in the query parameter `question`, and the filename should be specified in the query parameter `filename`.

3. Get Chunks:
   - **Route**: `http://localhost:5000/chunks?filename=userId__chatId`
   - **Method**: GET
   - **Query**: `?filename=userId__chatId`
   - **Description**: This API retrieves the chunks of data from the specified PDF file. The chunks are stored in JSON files and can be accessed using the filename specified in the query parameter `filename`.

4. Delete PDF:
   - **Route**: `http://localhost:5000/delete?filename=userId__chatId`
   - **Method**: POST
   - **Query**: `?filename=userId__chatId`
   - **Description**: This API allows the user to delete the specified PDF file. The filename of the PDF to be deleted should be specified in the query parameter `filename`.

## Prerequisites
- Python 3.6 or higher
- pip package manager

## Setup
1. Clone the repository to your local machine.
2. Create a virtual environment by running the following command:
   ```
   python3 -m venv myenv
   ```
3. Activate the virtual environment:
   - For Windows:
     ```
     myenv\Scripts\activate
     ```
   - For macOS/Linux:
     ```
     source myenv/bin/activate
     ```
4. Install the required packages by running the following command:
   ```
   pip install -r requirements.txt
   ```

## Usage
To run the application, execute the following command:
```
py .\app.py
```

Make sure to activate the virtual environment (`myenv`) before running the application to ensure the correct package versions are used.

## Note
- The application only has one endpoint running on Flask.
- PDF files are stored in JSON files, with each file named as "userId__chatId".
- Please refer to the `requirements.txt` file for a list of all required packages. You can install them all by running the command:
  ```
  pip install -r requirements.txt
  ```

## License
This project is licensed under the [MIT License](LICENSE).