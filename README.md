# HALIL-2DReacher: A Guide to Setup, Dataset Generation, and Training

## 1. Setup

### Install Required Libraries
To get started, ensure all the required libraries are installed. You can do this by running the following command in your terminal:

```bash
pip install -r requirements.txt
```

### Add Your OpenAI API Key
Ensure you have added your OpenAI API key to the environment. You can do this by setting it in your environment variables or by including it in your code where necessary. 

```bash
export OPENAI_API_KEY="your-key-here"
```

## 2. Generate a New Dataset

### Configure Dataset Parameters
To generate a new dataset of demonstrations, you need to configure the parameters in the `generate_demonstration_dataset.json` file located in the `src/config` directory. Modify the file to match your desired settings, such as the number of demonstrations, environment specifications, and any other relevant parameters.

### Run the Dataset Generation Script
Once you have configured the parameters, generate the dataset by running the following script:

```bash
python src/generate_demonstration_dataset.py
```

This will create a new dataset based on the configurations provided.

## 3. Run HALIL

### Configure Training Parameters
Before training the HALIL model, ensure the training configuration is set up correctly. Modify the `train_cp_ensomble.json` file located in the `src/config` directory to specify your training parameters
### Start Training
With your training configuration in place, start the training process by running the following command:

```bash
python src/train.py
```

This will initiate the training of the HALIL model using the specified configurations.
```
