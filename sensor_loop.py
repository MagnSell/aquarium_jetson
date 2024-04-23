import arduino_communication as ac
import database_communication as dc
import time
import pandas as pd
from utility import convert_sensor_data_to_dataframe
from datetime import datetime

def main():
    REFRESH_RATE = 5
    NUM_NODES = 2
    # Initialize the serial communication
    ser = ac.initialize_communication()
    conn = dc.initialize_conn()

    #Log
    logging = False
    log_df = pd.DataFrame()
    
    # Main Loop
    try:
        while True:
            # Receive data from the Arduino
            arduino_data = ac.receive_arduino_communication(ser)
            if arduino_data:
                node_measurements = convert_sensor_data_to_dataframe(arduino_data, NUM_NODES)
                dc.upsert_node_measurements(conn, node_measurements)
                if logging:
                    log_df =pd.concat([log_df,node_measurements],ignore_index=True)
            
            time.sleep(REFRESH_RATE)

    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        # Save the log
        if logging and not log_df.empty:
            datetime_string = datetime.now().strftime("%d%m%Y_%H%M%S")
            log_name = "logs/log_"+datetime_string+".csv"
            log_df.to_csv(log_name, index=False)

        # Close all communications
        ac.close_communication(ser)
        dc.close_conn(conn)

# Delete all the measurements from the database for running experiments and clearing old data
def delete_measurements():
    conn = dc.initialize_conn()
    dc.delete_all_node_measurements(conn)
    dc.close_conn(conn)

if __name__ == "__main__":
    #delete_measurements()
    main()
