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

2. Install the newest release from the [releases tab](https://github.com/skier233/nsfw_ai_model_server/releases) in this GitHub repository.

3. Unzip the contents of the release and place them where you want to run the server from.

4. Download any models you want to use from patreon and paste the contents of that zip file into the directory you just created.

5. Run the following command to create a new Conda environment:
    ```bash
    conda env create -f environment-{osname}.yml
    ```
    Replace `{osname}` with `windows` or `linux` based on your operating system.

6. Activate the Conda environment:
    ```bash
    conda activate ai_model_server
    ```

7. Start the server:
    ```bash
    python server.py
    ```
8. On the first time running the server, it will give you a link to a form to fill out and a code. Fill out that form with the provided code, license name, and your Patreon username.

9. You will receive a license file within 24 hours which you should put inside the `models` folder.

10. Run the server again:
    ```bash
    python server.py
    ```
11. To test the server, you can run the example client from the `example_client` folder.

## Updating Instructions

To update from a previous version, copy the contents of the zip file onto your existing directory, and then run update.ps1 or update.sh to upgrade your environment. 
