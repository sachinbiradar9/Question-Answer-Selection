from http.server import HTTPServer, BaseHTTPRequestHandler, SimpleHTTPRequestHandler
import json
from urllib.parse import parse_qs
from qa import test

class S(SimpleHTTPRequestHandler):

    def do_POST(self):
        print( "incomming http: ", self.path )
        content_length = int(self.headers['Content-Length']) # <--- Gets the size of data
        post_data = self.rfile.read(content_length).decode("utf-8") # <--- Gets the data itself
        post_data = json.loads(post_data)
        question = post_data['question']
        answers = post_data['answers'].split('||')
        response = answers[test(question, answers)]
        self.send_response(200)
        self.wfile.write(response.encode())

def run(server_class=HTTPServer, handler_class=S, port=80):
    server_address = ('', port)
    httpd = server_class(server_address, handler_class)
    print ('Starting httpd...')
    httpd.serve_forever()

if __name__ == "__main__":
    from sys import argv

    if len(argv) == 2:
        run(port=int(argv[1]))
    else:
        run()
