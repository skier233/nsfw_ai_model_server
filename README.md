# NSFW AI Model Server

This repository is designed for hosting a local HTTP server to access state-of-the-art NSFW AI models created by [Skier](https://www.patreon.com/Skier). You will need to be a patron to get access to these models to be able to use them with this repository.

The code here is built to be both fast, easy to install, and extendable.

## Limitations

Before becoming a patron, please be aware of the following limitations:

- As per the license, this code and associated models can only be used for personal local uses. For any for-profit use-cases or use-cases that need to be on the internet, [please contact](https://discord.gg/EvYbZBf) me for licensing options.

- Only NVIDIA GPUs are supported. Any Nvidia GPUs older than NVIDIA1080 will likely not work. Support for AMD GPUs is not planned.

- CPU is supported but it will be much slower than running on GPU.

- The nature of running machine learning models is very complex and requires everything to go precisely right to work smoothly. I've worked to make the installation process and AI model predictions as easy to use as possible but please understand that due to many people on different computers with different graphics cards and many other factors, there is a possibility you will run into issues. I will be here to help as best as I can to work through any of those issues, but I cannot guarantee that the models will be able to run on your computer.

- There are currently no tools built to make use of the predictions that this server outputs so non-developers may not gain the most value from this project yet until some of those tools are created to consume this data in meaningful ways.

## Installation Instructions

1. Install Conda: You can download and install Conda from [here](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html). Follow the instructions based on your operating system.

2. Download the newest release from the [releases tab](https://github.com/skier233/nsfw_ai_model_server/releases) in this GitHub repository.

3. Unzip the contents of the release and place them where you want to run the server from.

4. Download any models you want to use from patreon and paste the models folder from that zip file into the top level directory you just created.

5. Run (windows) `.\install.ps1` or (linux) `source .\install.sh`
   
6. On the first time running the server, it will open a browser window to login with patreon to get your license.

7. To test the server, you can run the example client from the `example_client` folder or proceed with installing the stash plugin.

## Updating Instructions

To update from a previous version run (windows) `.\update.ps1` or (linux) `source .\update.sh`

## Optional: Using Docker

### Prerequisites

1. Ensure Docker is installed on your system. You can download and install Docker from [here](https://docs.docker.com/get-docker/).

2. For GPU support, ensure you have the NVIDIA Container Toolkit installed. Follow the steps below to install it:

   - **Ubuntu:**

     ```sh
     sudo apt-get update
     sudo apt-get install -y nvidia-container-toolkit
     sudo systemctl restart docker
     ```

   - **Windows:**

     Follow the instructions [here](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) to set up the NVIDIA Container Toolkit on Windows.

### Building the Docker Container

1. Navigate to the directory where you unzipped the release.

2. Build the Docker image using the following command:

   ```sh
   docker build -t ai_model_server .
3. Modify the docker-compose.yml in the directory and modify this section with any paths to your content you want the ai server to be able to tag:
   ```sh
      - type: bind
        source: C:/Example/Media/Images/Folder
        target: /media/images/folder
        read_only: true
      - type: bind
        source: C:/Example/Media/Images/Folder2
        target: /media/images/folder2
        read_only: true
     ```
4. run `docker compose up` to start the server.
5. The first time you run the server, it'll try to authenticate with a browser and fail (since it can't start a browser in docker) and give you a form to fill out to request a license.
      After submitting the form you'll receive the license over patreon dms. Put the license file in your models folder and run `docker compose up` again.
6. The server will expect paths in the format of the `target` format above. If you send paths to the server that are paths from the host operating system it will not be able to see them. If you're using the official stash plugin, you can use the new path_mutation value in the config.py file in the plugin directory to mutate paths from stash that are sent to the server. If stash is also running in a docker container then you can use the same paths for the target in step 3 as in the stash container and then mutation will not be needed. If stash is not running in a docker container then you'll want to add each path you defined above to the path_mutation dictionary like so:
   ```sh
   path_mutation = {"C:/Example/Media/Images/Folder": "/media/images/folder", "C:/Example/Media/Images/Folder2", "/media/images/folder2"}
   ```
 
### Docker Updating Instructions
To update from a previous version run (windows) `.\update.ps1` or (linux) `source .\update.sh`.
Then, rebuild the docker container again using step 2 above.
