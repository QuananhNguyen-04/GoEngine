import os
import requests
from bs4 import BeautifulSoup

# Step 1: Provide the URL
url = 'http://gokifu.com/?p='  # Replace with the actual URL containing the player blocks

# Step 2: Fetch the HTML content
for i in range(30):
    page_url = url + str(i)
    response = requests.get(page_url)
    html_content = response.text
    # Step 3: Parse the HTML with Beautiful Soup
    soup = BeautifulSoup(html_content, 'html.parser')
    # Step 4: Find all the player blocks and extract .sgf links
    player_blocks = soup.find_all('div', class_='cblock_3')
    sgf_links = []
    # base_url = 'http://gokifu.com'  # Base URL to prepend to relative links
    # print(player_blocks)
    for block in player_blocks:
        game_type_blocks = block.find_all('div', class_='game_type')  # Find the first anchor without a title (which is the .sgf link)
        # print(game_type_blocks)
        for game_type_block in game_type_blocks:
            sgf_tag = game_type_block.find_all('a', href=True)
            # print("sgf_tag", sgf_tag)
            tag = sgf_tag[1]
            # print(tag['href'])
            if tag and tag['href'].endswith('.sgf'):
                sgf_links.append(tag['href'])


    # print('First 5 sgf links:')
    # for i, link in enumerate(sgf_links[:5]):
    #     print(link)
    #     if i < 4:
    #         print('\n')

    # assert False
    # Step 5: Download the .sgf files
    save_directory = './sgf_files'  # Directory to save the files

    # Create the directory if it doesn't exist
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    # assert False
    for link in sgf_links:
        file_name = os.path.join(save_directory, os.path.basename(link))

        if os.path.exists(file_name):
            print(f"File already exists: {file_name}")
            continue  # Skip downloading this file
        

        response = requests.get(link)
        
        if response.status_code == 200:  # Check if the request was successful
            with open(file_name, 'wb') as file:
                file.write(response.content)
            print(f"Downloaded: {file_name}")
        else:
            print(f"Failed to download: {link}")
