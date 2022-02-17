import os
from source.main import run

# Define the entry point for hooking data

# ! DO NOT MODIFY - ORGANIZER SPECIFIC
# <<<<<<<<<<<<<<<<<<<<<<<<<
DOCKER_ENTRY = {
    "input_dir": "/input/",
    "output_dir": "/output/",
    "user_data_dir": "/opt/algorithm/data/"
}
# >>>>>>>>>>>>>>>>>>>>>>>>>

# ! USER SPECIFIC
# <<<<<<<<<<<<<<<<<<<<<<<<<
LOCAL_ENTRY = {
    "input_dir": "/home/ll610/Onepiece/code_cs138/github/CoNIC-1/local_test/mha/",
    "output_dir": "/home/ll610/Onepiece/code_cs138/github/CoNIC-1/local_test/output/",
    "user_data_dir": "/home/ll610/Onepiece/code_cs138/github/CoNIC-1/conic_baseline/data/"
}
# >>>>>>>>>>>>>>>>>>>>>>>>>

# We have this parameter to adapt the paths between local execution
# and execution in docker. You can use this flag to switch between these two modes.

EXECUTE_IN_DOCKER = False


if __name__ == "__main__":
    print(f"\nWorking Directory: {os.getcwd()}")
    print("\n>>>>>>>>>>>>>>>>> Start User Script\n")
    # Trigger the inference in the `source` directory
    ENTRY = DOCKER_ENTRY if EXECUTE_IN_DOCKER else LOCAL_ENTRY
    run(**ENTRY)
    print("\n>>>>>>>>>>>>>>>>> End User Script\n")
