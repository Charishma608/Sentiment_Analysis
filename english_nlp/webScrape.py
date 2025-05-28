import requests
import pandas as pd
from langdetect import detect
import csv

API_KEY = 'AIzaSyDqRm40yiDTJmp-ffi1_85oTooeUFv2_00'

def fetch_youtube_comments(video_id, max_comments=2000):
    comments = []
    url = f"https://www.googleapis.com/youtube/v3/commentThreads?key={API_KEY}&textFormat=plainText&part=snippet&videoId={video_id}&maxResults=100"
    
    while url and len(comments) < max_comments:
        response = requests.get(url)
        
        if response.status_code == 200:
            data = response.json()
            for item in data['items']:
                comment_text = item['snippet']['topLevelComment']['snippet']['textOriginal']
                comment_text = comment_text.replace('\n', ' ').replace('\r', ' ')
                
                try:
                    if detect(comment_text) == 'en':
                        comments.append(comment_text)
                        if len(comments) >= max_comments:
                            break
                except:
                    continue
            
            next_page_token = data.get('nextPageToken')
            if next_page_token:
                url = (f"https://www.googleapis.com/youtube/v3/commentThreads?key={API_KEY}"
                       f"&textFormat=plainText&part=snippet&videoId={video_id}&maxResults=100"
                       f"&pageToken={next_page_token}")
            else:
                url = None
        else:
            print(f"Error fetching comments: {response.status_code}")
            print(f"Response Content: {response.text}")
            break
    
    return comments

video_id = 'zsO1Oayp7kg'
max_comments = 100
all_comments = []

comments = fetch_youtube_comments(video_id, max_comments)
all_comments.extend(comments)

unique_comments = list(set(all_comments))

df = pd.DataFrame(unique_comments, columns=['comment'])
df['comment'] = df['comment'].apply(lambda x: x.replace(',', ''))
df.to_csv('Scrapped.csv', index=False, quoting=csv.QUOTE_ALL)

print("Comments saved to Scrapped.csv")
