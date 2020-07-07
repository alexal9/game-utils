from selenium.webdriver.support.ui import Select
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup as bs
import time
import pickle
import sys

def update(): 
    start = time.time()
    # sys.setrecursionlimit(10000)

    try:
        driver = webdriver.Chrome()
        driver.get("""https://swarfarm.com/bestiary""")

        WebDriverWait(driver, 10).until(
            lambda d: d.execute_script("return document.readyState") == "complete"
        )

        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "td"))
        )

        webpage = driver.find_element_by_css_selector("html")
        soup = bs(webpage.get_attribute('innerHTML'), features='lxml')

        num_buttons = len( soup.select('button[data-page]') ) // 2 - 1

        # store string representation to prevent issues with pickling
        results = soup.select('td')
        bestiary = [ repr(tag) for tag in results ]
        prev_mon = results[1].get_text()
        print("bestiary size", len(bestiary))

        for page in range(2, num_buttons + 1):
            button = driver.find_element_by_css_selector("button[data-page='{}']".format(page))
            button.click()
            print("loading page", page)

            while True:
                time.sleep(7)
                webpage = driver.find_element_by_css_selector("html")
                soup = bs(webpage.get_attribute('innerHTML'), features='lxml')

                results = soup.select('td')
                if results[1].get_text() != prev_mon:
                    print("new page loaded", page, prev_mon)
                    break

            prev_mon = results[1].get_text()
            bestiary.extend([ repr(tag) for tag in soup.select('td') ])
            print("bestiary size", len(bestiary))

        with open('bestiary.pickle','wb') as f:
            pickle.dump(bestiary, f)
    except Exception as e:
        print("error occured", e)
    finally:
        driver.quit()
        print("total time:", time.time() - start)

if __name__ == '__main__':
    update()
