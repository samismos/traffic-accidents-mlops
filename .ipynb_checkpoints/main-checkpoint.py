import mlrun

# Create or get the MLRun project
project = mlrun.get_or_create_project(name="traffic-accidents", context="./")

# Convert the Python script to a function
function = mlrun.code_to_function(
    name="simple-function", 
    kind="job",
    filename="script.py"
)


# Run the function
function.run(handler="entrypoint")  # Specify the function to run
