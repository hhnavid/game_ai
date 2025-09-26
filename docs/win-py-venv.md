* To create a Python 3.13 virtual environment on Windows 10, follow these steps:

    * Open a Command Prompt in the folder where you want your virtual environment. You can do this by navigating to the folder in File Explorer, typing cmd in the address bar, and pressing Enter.

    * Run the following command to create the virtual environment (replace myenv with your desired environment name):
        * $ python -m venv myenv
        * This will create a folder named myenv containing the isolated Python 3.13 environment.

    * Activate the virtual environment:

* activate env
    * If using Command Prompt:
        * myenv\Scripts\activate.bat
    * If using PowerShell:
        * .\myenv\Scripts\activate.ps1
    * __use this__ If using __Git Bash__ or similar:
        * <font color=blue>source myenv/Scripts/activate</font>
* Once activated, your prompt will show the environment name, indicating that you are now working inside the virtual environment
* __to active an existing python virtual env in vscode terminal__
    * in vscode terminal, change the execution policy of the power shell so that scripts can be executed:       
      * $ Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy Bypass
    * now in vscode terminal run the command below to activate the python virtual env named venv:
        * $ .\venv\Scripts\activate 