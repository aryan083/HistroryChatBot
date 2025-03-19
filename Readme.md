# Generative Text Comprehension For Question Answering on History Topics

This console application is a Generative question answering app, which uses a knowledge based architecture that allows it to answer concepts that require knowledge of question being asked and and returns a short and clear answer

## Feature

**Wide knowledge:** it has knowledge of 100000 history topics which allows it to answer on an array of topics 

**Coherent Answer:** outputs a clear and short answer about the given topic

## Installation

Make sure you are connected to internet for the whole process.

Installing required packages   
```
pip install -r packages.txt
```

## Environment Variables 
To run this project, you will need to add the following environment variables to your .env file from pinecone console

`API_KEY` 

`ENVIRONMENT`

## Running Tests

To run tests, run the following command 

```
python script.py
```
> For the first time it will need to download the model so be patient

From the option choose `1` to ask question

```
1
```

Then ask any history related question 

### Example questions
- *When was The Bishop Wand Church of England School founded and who is it named after?*
- *How did Viktoria progress through the pyramid levels in football?*
- *Until 719, what was the status of Suyab in relation to the Anxi Protectorate?*
- *Which local rock label did Star Records sign a 3-year license agreement with?*