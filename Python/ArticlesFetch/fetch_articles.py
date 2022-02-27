import http.client
import json

conn = http.client.HTTPSConnection("newscatcher.p.rapidapi.com")

headers = {
    'x-rapidapi-host': "newscatcher.p.rapidapi.com",
    'x-rapidapi-key': "1003d62a71msh23a608e75d952c1p19542fjsn3d3e40adc5d0"
}

conn.request("GET", "/v1/latest_headlines?lang=en&media=True", headers=headers)

res = conn.getresponse()
data = res.read()

news = json.loads(data)['articles']

for piece in news:
    print(piece['summary'])