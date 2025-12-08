"""
twitter-politic-scraper.py

Simple script using selenium and undetected-chromedriver to scrape political tweets from Twitter/X.
"""

import argparse
import pandas as pd
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
import time
import undetected_chromedriver as uc

#ensures a fresh profile to keep from twitter bot detection ruining scraper
USER_DATA_DIR = r"C:/temp/ucprofile"  

def find(elem, xpath):
    """
    Finds an element's text content using the provided XPath.
    
    :param elem: selenium WebElement representing the parent element
    :param xpath: XPath string to locate the child element
    :returns: Text content of the found element or empty string if not found
    """
    try:
        return elem.find_element(By.XPATH, xpath).text
    except:
        return ""

def attr(elem, xpath, attr):
    """
    Finds an attribute value of an element using the provided XPath.
    
    :param elem: selenium WebElement representing the parent element
    :param xpath: XPath string to locate the child element
    :param attr: Attribute name to retrieve
    :returns: Attribute value of the found element or empty string if not found
    """
    try:
        return elem.find_element(By.XPATH, xpath).get_attribute(attr)
    except:
        return ""

def is_reply_element(tweet_element):
    """
    Determines if a tweet element is a reply.
    
    :param tweet_element: selenium WebElement representing the tweet
    :returns: True if the tweet is a reply, False otherwise
    """
    try:
        tweet_element.find_element(By.XPATH, ".//*[@data-testid='replying-to-context']")
        return True
    except:
        return False

def find_tweet_text(tweet_element):
    """
    Finds the text content of a tweet element.
    
    :param tweet_element: selenium WebElement representing the tweet
    :returns: text content of the tweet
    """
    try:
        text_element = tweet_element.find_element(By.XPATH, ".//div[@data-testid='tweetText']")
        return text_element.text.replace("\n", " ").strip()
    except:
        return ""

def scrape_twitter(output_file, search_query, amount, scroll_pause):
    """
    Docstring for scrape_twitter
    
    :param output_file: Description
    :param search_query: Description
    :param amount: Description
    :param scroll_pause: Description
    """
    print("Launching undetected Chrome...")
    options = uc.ChromeOptions()
    options.add_argument("--no-first-run")
    options.add_argument("--no-service-autorun")
    options.add_argument(f"--user-data-dir={USER_DATA_DIR}")

    driver = uc.Chrome(options=options, version_main=142)
    driver.set_window_size(1100, 900)

    driver.get("https://twitter.com/login")
    time.sleep(4)
    input("\nLog in to Twitter/X manually, then press ENTER here...\n")

    search_url = f"https://twitter.com/search?q={search_query}&src=typed_query&f=live"
    driver.get(search_url)
    time.sleep(4)

    print(f"Scraping tweets for query: {search_query}... (ONLY original posts)")
    rows = []
    seen_texts = set()
    last_count = 0

    while len(rows) < amount:
        tweets = driver.find_elements(By.XPATH, "//article[@data-testid='tweet']")

        for t in tweets:
            try:
                #skip replies (lacks context and therefore holds no meaning
                if is_reply_element(t):
                    continue

                text = find_tweet_text(t) 
                
                # Validation checks for text
                if not text or text in seen_texts:
                    continue

                seen_texts.add(text)

                username = find(t, ".//span[contains(text(), '@')]")
                timestamp = attr(t, ".//time", "datetime")
                replies = find(t, ".//*[@data-testid='reply']")
                retweets = find(t, ".//*[@data-testid='retweet']")
                likes = find(t, ".//*[@data-testid='like']")
                views = find(t, ".//*[@data-testid='view']")

                rows.append({
                    "text": text,
                    "username": username,
                    "timestamp": timestamp,
                    "replies": replies,
                    "retweets": retweets,
                    "likes": likes,
                    "views": views,
                    "platform": "Twitter/X",
                })

                print(f"[{len(rows)}] {text[:75]}...")

                if len(rows) >= amount:
                    break

            except:
                continue

        driver.find_element(By.TAG_NAME, "body").send_keys(Keys.END)
        time.sleep(scroll_pause)

        #pause scrolling if no new posts found
        if len(rows) == last_count:
            print(f"Possibly rate-limited or reached bottom, pausing {scroll_pause}s…")
            time.sleep(scroll_pause)
        last_count = len(rows)

    driver.quit()

    df = pd.DataFrame(rows)
    df.to_csv(output_file, index=False, encoding="utf-8-sig")
    print(f"\Saved {len(rows)} posts to {output_file}")

if __name__ == "__main__":
    """
    Main entry point for twitter-politic-scraper.py script.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_csv", required=True)
    parser.add_argument("--search_query", type=str, default="politics")
    parser.add_argument("--amount", type=int, default=500)
    parser.add_argument("--scroll_pause", type=int, default=4)
    args = parser.parse_args()

    output_file = args.output_csv
    search_query = args.search_query
    amount = args.amount
    scroll_pause = args.scroll_pause

    scrape_twitter(output_file, search_query, amount, scroll_pause)