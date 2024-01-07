import urllib
import time
import feedparser

################## DIRECT LINK TO DOWNLOAD PDF #####################
https://arxiv.org/pdf/{id}.pdf: #Direct link to download the PDF. Substitute {id}.pdf with the id from the JSON extract: '0704.0001.pdf'

# Base api query url
base_url = 'http://export.arxiv.org/api/query?'

# Search parameters
search_query = 'all:machine learning&sortBy=lastUpdatedDate&sortOrder=ascending' # search for 'machine learning' in all fields
start = 0                       # start at the first result
total_results = 20              # want 20 total results
results_per_iteration = 5       # 5 results at a time
wait_time = 3                   # number of seconds to wait beetween calls

print('Searching arXiv for %s' % search_query)

for i in range(start, total_results, results_per_iteration):
    
    print("Results %i - %i" % (i,i+results_per_iteration))
    
    query = 'search_query=%s&start=%i&max_results=%i' % (search_query,
                                                         i,
                                                        results_per_iteration)

    # perform a GET request using the base_url and query
    response = urllib.urlopen(base_url+query).read()

    # parse the response using feedparser
    feed = feedparser.parse(response)

    # Run through each entry, and print out information
    for entry in feed.entries:
        print('arxiv-id: %s' % entry.id.split('/abs/')[-1])
        print('Title:  %s' % entry.title)
        # feedparser v4.1 only grabs the first author
        print('First Author:  %s' % entry.author)
    
    # Enforce waiting time so that API doesn't get overwhelmed
    print('Sleeping for %i seconds' % wait_time)
    time.sleep(wait_time)