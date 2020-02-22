import json
import urllib.parse
from http.server import BaseHTTPRequestHandler, HTTPServer
import cgi
from cgi import parse_header, parse_multipart

import predict_emotion

class GetHandler(BaseHTTPRequestHandler):

    def do_POST(self):
        ctype, pdict = cgi.parse_header(self.headers.get('content-type')) #
        if ctype == 'multipart/form-data':
            postvars = cgi.parse_multipart(self.rfile, pdict)
        elif ctype == 'application/x-www-form-urlencoded':
            length = int(self.headers.get('content-length'))
            postvars = urllib.parse.parse_qs(self.rfile.read(length), keep_blank_values=1)
        else:
            postvars = {}
            
        print (postvars)
        
        data = predict_emotion.loaddata()
        res = predict_emotion.predictEmotion(data, postvars)
        
        pred = json.dumps(res) 
        self.wfile.write(pred.encode())

        #self.send_response(200)
        #self.end_headers()

        return
        
if __name__ == '__main__':
    server = HTTPServer(('localhost', 8080), GetHandler)
    print ('Starting server, use <Ctrl-C> to stop')
    server.serve_forever()


'''
To run this:
1. go to a command line, run python webserver.python
2. go to another command line, type:
curl -d "{ \"headline\": \"headline_content", \"summary\": \"summary_content\", \"worker_id\": 1 }" http://localhost:8080

The entire project can be found:
C:\Users\yluo\Downloads\homework.json.gz
'''