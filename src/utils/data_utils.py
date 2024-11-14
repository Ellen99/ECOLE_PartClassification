'''Utility functions for loading and saving data'''
import os
import pickle

# Save train_concept_parts to a file
def save_concept_hierarchy(concept_parts, filename):
    '''Save concept_parts to a file'''
    checkpoint_dir = os.path.dirname(filename)
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    with open(filename, 'wb') as f:
        pickle.dump(concept_parts, f)
    print(f"Concept hierarchy saved to {filename}")

# Load concept_parts from a file
def load_concept_hierarchy(filename):
    '''Load concept_parts from a file'''
    with open(filename, 'rb') as f:
        concept_hierarchy = pickle.load(f)
    print(f"Concept hierarchy loaded from {filename}")
    return concept_hierarchy