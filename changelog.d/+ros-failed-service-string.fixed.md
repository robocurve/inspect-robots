**ROS plugin:** Accept rosbridge's failed `service_response`, whose `values`
is the error string rather than an object. A failed `reset_service` call now
raises `service_failed` with rosbridge's message instead of an `invalid_frame`
error that also stopped the receive thread.
