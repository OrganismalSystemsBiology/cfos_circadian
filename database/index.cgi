#!/home/webpark2445/local/bin/python3

import cgitb
cgitb.enable()

from wsgiref.handlers import CGIHandler

from sys import path

path.insert(0, '/home/webpark2445/www/database/')
from app.app import app

class ProxyFix(object):
  def __init__(self, app):
      self.app = app
  def __call__(self, environ, start_response):
      environ['SERVER_NAME'] = "webpark2445.sakura.ne.jp"
      environ['SERVER_PORT'] = "80"
      # REQUEST_METHODの設定を行う
      if 'REQUEST_METHOD' not in environ:
            environ['REQUEST_METHOD'] = 'GET'  # デフォルトはGET
        
        # POSTリクエストの場合
      if environ['REQUEST_METHOD'] == 'POST':
            environ['REQUEST_METHOD'] = 'POST'  # 上書きは必要ないが、明示的に設定      #environ['REQUEST_METHOD'] = "GET"
      environ['SCRIPT_NAME'] = "/database/app"
      # if 'PATH_INFO' not in environ or environ['PATH_INFO'] == '':
      #       environ['PATH_INFO'] = '/'

      # PATH_INFOを設定
      # if environ.get('PATH_INFO') in ['/database/', '/database']:
      #     environ['PATH_INFO'] = '/login'  # /loginにリダイレクト
      #     return self.app(environ, start_response)
      
    #   environ['PATH_INFO'] = "/"
    #   environ['QUERY_STRING'] = ""
      environ['SERVER_PROTOCOL'] = "HTTP/1.1"
      return self.app(environ, start_response)
if __name__ == '__main__':
   app.wsgi_app = ProxyFix(app.wsgi_app)
   CGIHandler().run(app)
