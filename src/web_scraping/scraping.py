from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
import pandas as pd
import logging

logging.basicConfig(filename='logs/scraping.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def setup_driver():
    """Starts and configures the WebDriver."""
    driver = webdriver.Chrome()
    driver.set_page_load_timeout(120)
    driver.set_script_timeout(120)
    return driver

def scrape_car_data(driver, url, max_pages=22):
    """
    Scrapes car data from Turbo.az for a specific brand.
    """
    wait = WebDriverWait(driver, 150)
    data = {
        "Price": [], "Make": [], "Model": [], "Year": [],
        "Color": [], "Engine": [], "Kilometer": [], "Transmission": [], "New": []
    }

    try:
        for page in range(1, max_pages + 1):
            # Page loading mechanism
            max_retries = 3  # Maximum number of retries for page loading
            for attempt in range(max_retries):
                try:
                    driver.get(url.format(page))
                    time.sleep(5)
                    break
                except Exception as error:
                    logging.error(f"Failed to load page {page}. Retrying ({attempt + 1}/{max_retries})... Error: {error}")
                    if attempt == max_retries - 1:
                        raise error

            # Listing fetching mechanism
            max_retries = 3 
            for attempt in range(max_retries):
                try:
                    listings = wait.until(EC.visibility_of_all_elements_located((By.CLASS_NAME, "products-i__link")))
                    break
                except Exception as error:
                    logging.error(f"Failed to fetch listings on page {page}. Retrying ({attempt + 1}/{max_retries})... Error: {error}")
                    if attempt == max_retries - 1:
                        raise error

            # Scraping data for each listing
            for listing in listings:
                try:
                    wait.until(EC.element_to_be_clickable(listing))
                    listing.click()
                    driver.switch_to.window(driver.window_handles[-1])  # Switch to the new tab

                    # Data extraction process
                    max_retries = 3
                    for attempt in range(max_retries):
                        try:
                            data["Price"].append(wait.until(EC.presence_of_element_located((By.XPATH, "//div[@class='product-price__i product-price__i--bold']"))).text.strip())
                            data["Make"].append(wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "label[for='ad_make_id'] + span"))).text.strip())
                            data["Model"].append(wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "label[for='ad_model'] + span"))).text.strip())
                            data["Year"].append(wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "label[for='ad_reg_year'] + span"))).text.strip())
                            data["Color"].append(wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "label[for='ad_color'] + span"))).text.strip())
                            data["Engine"].append(wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "label[for='ad_engine_volume'] + span"))).text.strip())
                            data["Kilometer"].append(wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "label[for='ad_mileage'] + span"))).text.strip())
                            data["Transmission"].append(wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "label[for='ad_transmission'] + span"))).text.strip())
                            data["New"].append(wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "label[for='ad_new'] + span"))).text.strip())
                            break
                        except Exception as error:
                            logging.error(f"Failed to extract data. Retrying ({attempt + 1}/{max_retries})... Error: {error}")
                            if attempt == max_retries - 1:
                                raise error

                    driver.close()
                    driver.switch_to.window(driver.window_handles[0])  # Switch back to the main tab
                except Exception as error:
                    logging.error(f"Error extracting data from a listing: {error}")
                    driver.close()
                    driver.switch_to.window(driver.window_handles[0])

    except Exception as error:
        logging.critical(f"Critical error occurred: {error}")
    finally:
        driver.quit()

    return pd.DataFrame(data)

def save_to_csv(df, filename):
    """Saves the DataFrame as a CSV file."""
    df.to_csv(filename, index=False)

def validate_data(data):
    """Validates the scraped data."""
    for key, values in data.items():
        if not all(values):
            logging.warning(f"Some values in {key} are empty or invalid.")

if __name__ == "__main__":
    brands = {
        "Mercedes": "https://turbo.az/autos?page={}&q%5Bmake%5D%5B%5D=4&q%5Byear_from%5D=2019",
        "Hyundai": "https://turbo.az/autos?page={}&q%5Bmake%5D%5B%5D=1&q%5Byear_from%5D=2019",
        "Kia": "https://turbo.az/autos?page={}&q%5Bmake%5D%5B%5D=8&q%5Byear_from%5D=2019",
        "Bmw": "https://turbo.az/autos?page={}&q%5Bmake%5D%5B%5D=3&q%5Byear_from%5D=2019"
    }

    driver = setup_driver()

    # Scrape and save data for each brand
    for brand, url in brands.items():
        print(f"Scraping data for {brand}...")
        df = scrape_car_data(driver, url)
        validate_data(df.to_dict('list')) 
        save_to_csv(df, f"data/raw/{brand.lower()}.csv")
        print(f"Data for {brand} saved to data/raw/{brand.lower()}.csv")

    driver.quit()