#!/usr/bin/env python3
from dataclasses import dataclass
from typing import Dict, List, Tuple
import pickle, re, os, sys, time, threading, webbrowser, json, difflib, contextlib
from tinygrad.helpers import getenv
from tinygrad.ops import TrackedRewriteContext, UOp, UOps
from tinygrad.engine.graph import uops_colors, word_wrap
from http.server import HTTPServer, BaseHTTPRequestHandler

stop_reloader = threading.Event()
def reloader():
  mtime = os.stat(__file__).st_mtime
  while not stop_reloader.is_set():
    if mtime != os.stat(__file__).st_mtime:
      print("reloading server...")
      os.execv(sys.executable, [sys.executable] + sys.argv)
    time.sleep(0.1)

def uop_to_json(x:UOp) -> Dict[int, Tuple[str, str, List[int], str, str]]:
  assert isinstance(x, UOp)
  graph: Dict[int, Tuple[str, str, List[int], str, str]] = {}
  for u in x.sparents:
    label = f"{str(u.op)[5:]}{(' '+word_wrap(str(u.arg).replace(':', ''))) if u.arg is not None else ''}\n{str(u.dtype)}"
    if getenv("WITH_SHAPE"):
      with contextlib.suppress(Exception): # if the UOp is indexed already it's fine
        if u.st is not None: label += f"\n{u.st.shape}"
    graph[id(u)] = (label, str(u.dtype), [id(x) for x in u.src], str(u.arg), uops_colors.get(u.op, "#ffffff"))
  return graph

@dataclass(frozen=True)
class UOpRet:
  loc: str                           # location that called graph_rewrite
  uops: List[UOp]                    # snapshot of the entire AST after each rewrite
  diffs: List[Tuple[str, List[str]]] # the diffs for each rewrite
  extra: List[List[str]]             # these become code blocks in the UI

def replace_uop(base:UOp, prev:UOp, new:UOp, cache:Dict[bytes, UOp]) -> UOp:
  if (found:=cache.get(base.key)): return found
  if base.key == prev.key: ret = new
  else:
    new_srcs = tuple(replace_uop(x, prev, new, cache) for x in base.src)
    ret = UOp(base.op, base.dtype, new_srcs, base.arg) if new_srcs != base.src else base
  cache[base.key] = ret
  return ret

def create_graph(ctx:TrackedRewriteContext) -> UOpRet:
  uops: List[UOp] = [ctx.sink]
  diffs: List[Tuple[str, List[str]]] = []
  extra: List[List[str]] = [[str(ctx.sink)]]
  for (first, rewritten, pattern) in ctx.rewrites:
    # if the sink was replaced, we have to replace the entire graph, otherwise just replace the parent
    new_sink = rewritten if first.op is UOps.SINK else replace_uop(uops[-1], first, rewritten, {})
    # TODO: sometimes it hits a ctx and can't find any UOp to replace
    #if new_sink is uops[-1]: continue
    diffs.append((pattern, list(difflib.unified_diff(str(first).splitlines(), str(rewritten).splitlines()))))
    assert new_sink.op is UOps.SINK
    uops.append(new_sink)
    extra.append([str(new_sink)])
  return UOpRet(ctx.loc, uops, diffs, extra)

class Handler(BaseHTTPRequestHandler):
  def do_GET(self):
    if self.path == "/favicon.svg":
      self.send_response(200)
      self.send_header("Content-type", "image/svg+xml")
      self.end_headers()
      with open(os.path.join(os.path.dirname(__file__), "favicon.svg"), "rb") as f:
        ret = f.read()
    if self.path == "/":
      self.send_response(200)
      self.send_header("Content-type", "text/html")
      self.end_headers()
      with open(os.path.join(os.path.dirname(__file__), "index.html"), "rb") as f:
        ret = f.read()
    elif re.search(r'/\d+', self.path):
      self.send_response(200)
      self.send_header("Content-type", "application/json")
      self.end_headers()
      with open("/tmp/rewrites.pkl", "rb") as f: contexts: List[TrackedRewriteContext] = pickle.load(f)
      rest = [x.loc for x in contexts]
      g = create_graph(contexts[int(self.path.split("/")[-1])])
      ret = json.dumps(({"loc": g.loc, "graphs": list(map(uop_to_json, g.uops)), "diffs": g.diffs, "extra": g.extra}, rest)).encode()
    else:
      self.send_response(404)
      ret = b""
    return self.wfile.write(ret)

BROWSER = getenv("BROWSER", 1)
def main():
  try:
    st = time.perf_counter()
    reloader_thread = threading.Thread(target=reloader)
    reloader_thread.start()
    print("serving at port 8000")
    server_thread = threading.Thread(target=HTTPServer(('', 8000), Handler).serve_forever, daemon=True)
    server_thread.start()
    if BROWSER: webbrowser.open("http://localhost:8000")
    print(f"{(time.perf_counter()-st):.2f}s startup time")
    server_thread.join()
    reloader_thread.join()
  except KeyboardInterrupt:
    print("viz is shutting down...")
    stop_reloader.set()

if __name__ == "__main__":
  main()
