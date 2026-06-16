#!/usr/bin/env python
"""Serve uma PÁGINA ÚNICA (porta 8090) juntando, lado a lado: o robô do sim (laptop:8013)
e o OmniView/attention/RGB/depth (lcad232 via forward local 8014). Tudo numa tela só."""
import http.server, socketserver

PAGE = ("""<!doctype html><html><head><meta charset=utf-8>
<title>G1 — eval offline ao vivo (robô + attention)</title>
<style>
 html,body{margin:0;height:100%;background:#0b0b0d;color:#e8e8ea;font-family:system-ui,sans-serif;overflow:hidden}
 .top{height:34px;display:flex;align-items:center;gap:14px;padding:0 14px;background:#15151a;border-bottom:1px solid #26262c}
 .top b{color:#7db5ff} .top span{color:#888;font-size:12px}
 .grid{display:grid;grid-template-columns:46% 54%;height:calc(100% - 35px)}
 .cell{position:relative;border-right:1px solid #26262c;overflow:hidden}
 .lbl{position:absolute;top:8px;left:8px;z-index:2;background:#000a;padding:3px 9px;border-radius:6px;font-size:12px;color:#7db5ff}
 iframe{width:100%;height:100%;border:0;background:#000}
</style></head>
<body>
 <div class=top><b>G1 — eval offline AO VIVO</b>
   <span>VLA real (RTX 4090 / lcad232) vê o vídeo do dataset → controla o robô do simulador</span></div>
 <div class=grid>
   <div class=cell><div class=lbl>🤖 Robô no simulador (3ª pessoa)</div>
       <iframe src="http://localhost:8013/"></iframe></div>
   <div class=cell style="border-right:0"><div class=lbl>👁️ OmniView — RGB · attention · depth · tátil · juntas</div>
       <iframe src="http://localhost:8014/live.html"></iframe></div>
 </div>
</body></html>""").encode("utf-8")


class H(http.server.BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def do_GET(self):
        self.send_response(200); self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(PAGE))); self.end_headers(); self.wfile.write(PAGE)


socketserver.TCPServer.allow_reuse_address = True
with socketserver.TCPServer(("127.0.0.1", 8090), H) as srv:
    print("combined em http://localhost:8090", flush=True)
    srv.serve_forever()
