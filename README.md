

# Point Processes

This is a repository of all my python based point process code.

So far the data representation is built to match online learning for
Hawkes processes. The code for spatial PCIMs is being written.

The goal here is to have a general framework for working with
Poisson Processes.

My tasks I wish to have is prediction

The data sources I plan on using are crime and network traffic at the moment.

# Coding tasks

I need to verify everything is working. 
I want to perform sampling from the Hawkes processes.

# Problems to look at:

## Dataset: Crime
Question:
Do the districts in LA match a realistic description of the space
or will some other partitioning perform better?

### Question: Will PCIMs perform better at just trying to minimize a mean squared error?

The task for crime data set is going to be getting total counts over a week for
each crime type using the PCIM to break up the spatial component. This means
I should probably be using some other regression method.

### Question: Can I derive a method for just minimizing the error?

Should I care about pcims?

## Dataset: Network Dataset
### Question: Will point processes make a good method for intrusion detection?
For the IDS I want to build some sort of anomoly detection method.

I need to do more research into this field.
