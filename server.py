import os
from lib.server.server_manager import app
import lib.server.routes
import uvicorn

# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))
# Change the current working directory to the script's directory
os.chdir(script_dir)

if __name__ == "__main__":
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
