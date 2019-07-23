import socketio
import eventlet
import eventlet.wsgi as wsgi


class SimulatorAutoDrive:
    socket = None
    predictor = None

    def __init__(self, predictor):
        self.predictor = predictor
        self.socket = socketio.Server()
        self.socket.on("connect", self.__connect)
        self.socket.on("telemetry", self.__telemetry)

    def __telemetry(self, sid, data):
        steer, throttle = self.predictor(sid, data)
        print("Steering: {}, Throttle: {}".format(steer, throttle))
        self.__send(steer, throttle)

    def __connect(self, sid, environ):
        print("connected", sid)
        self.__send(0, 0)

    def run(self):
        wsgi.server(eventlet.listen(('', 4567)), socketio.WSGIApp(self.socket))

    def __send(self, steering, throttle):
        self.socket.emit(
            "steer",
            data={
                "steering_angle": str(steering),
                "throttle": str(throttle),
            },
            skip_sid=True
        )
