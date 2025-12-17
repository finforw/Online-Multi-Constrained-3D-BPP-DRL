import datetime
def get_time_range():
    current_timestamp = int(datetime.datetime.now().timestamp())
    future_date = datetime.datetime.now() + datetime.timedelta(days=7)
    future_timestamp = int(future_date.timestamp())
    return (current_timestamp, future_timestamp)
