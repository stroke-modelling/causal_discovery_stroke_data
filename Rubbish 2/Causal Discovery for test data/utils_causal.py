"""Functions to help with causal modelling."""
# Imports
import pandas as pd
from dowhy import CausalModel

# Functions
def make_causalmodel_object(
        nodes_labels, edges, treatment_str='Treatment', outcome_str='Outcome'
        ):
    gml_string = make_gml_string(nodes_labels, edges)
    # Make an empty DataFrame because the CausalModel expects a df:
    nodes = [n[0] for n in nodes_labels]
    df_data = pd.DataFrame(columns=nodes)  # empty
    # Instantiate the CausalModel 
    model = CausalModel(
        data=df_data,
        treatment=treatment_str,
        outcome=outcome_str,
        graph=gml_string
    )
    return model


def make_gml_string(nodes_labels, edges):
    """Make GML string. Contains each node and edge."""
    # Generate the GML graph
    gml_string = 'graph [directed 1\n'
    
    for (node, label) in nodes_labels:
        gml_string += f'\tnode [id "{node}" label "{label}"]\n'
    
    for edge in edges:
        gml_string += f'\tedge [source "{edge[0]}" target "{edge[1]}"]\n'
        
    gml_string += ']'
    return gml_string