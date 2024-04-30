import uuid
from datetime import datetime
from schema import Node_Measurement, Fish
import pandas as pd

def convert_sensor_data_to_dataframe(data, num_nodes):
    # Create a Node_Measurement object
    node_measurements = []
    for node_id in range(num_nodes):
        node_measurement = Node_Measurement(
            uuid = uuid.uuid4(),
            timestamp = datetime.now(),
            node_id=node_id,
            temperature=data["temperature"][node_id],
            pH=data["pH"][node_id],
            dissolved_oxygen=data["dissolved_oxygen"][node_id]
        )
        node_measurements.append(node_measurement)

    # Create a dataframe from the node measurements
    df = pd.DataFrame([nm.__dict__ for nm in node_measurements])
    return df

def create_fish_object(object,timestamp):
    # Create a Fish object
    fish = None
    try:
        fish = Fish(
            uuid=uuid.uuid4(),
            fish_id=object.id,
            timestamp=timestamp,
            x_position=object.position[0],
            y_position=object.position[1],
            z_position=object.position[2],
            x_velocity=object.velocity[0],
            y_velocity=object.velocity[1],
            z_velocity=object.velocity[2]
        )
    except Exception as e:
        print("Error: ", e)
        print("Object: ", object)
    return fish