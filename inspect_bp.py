'''
Python script to inspect .bp file and its variables
'''

import adios2

# Open the .bp file
with adios2.open("/root/shared/TEAM30_300.0_tree/B.bp", "r") as fh:
    for step in fh:
        print("Step:", step.current_step())
        for name in step.available_variables():
            print("Variable:", name)
            print("Metadata:", step.available_variables()[name])
