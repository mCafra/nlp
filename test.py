import requests

url = "https://textanalysis-text-summarization.p.rapidapi.com/text-summarizer"

payload = {
	"url": "http://en.wikipedia.org/wiki/Automatic_summarization",
	"text": "",
	"sentnum": 8
}
headers = {
	"content-type": "application/json",
	"X-RapidAPI-Key": "ea4c996351mshd7dbddb8d4bdcadp1c9b96jsn5fb925ef9845",
	"X-RapidAPI-Host": "textanalysis-text-summarization.p.rapidapi.com"
}

response = requests.request("POST", url, json=payload, headers=headers)

print(response.text)