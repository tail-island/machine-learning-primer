import csv
import requests

from bs4 import BeautifulSoup
from funcy import concat, first
from itertools import starmap
from time import sleep
from urllib.parse import urljoin


def get_soup(url):
    html = requests.get(url)
    html.encoding = 'UTF8'

    return BeautifulSoup(html.text, 'html.parser')


def get_row_urls(category_url):
    sleep(10)  # サイトに迷惑をかけないよう、スリープして10秒待ちます。

    for li in get_soup(category_url).select('div.bikeList li'):
        if li.select('div.currentModel'):  # 現行車のみとします
            yield urljoin(category_url, li.select_one('a').attrs['href'])


def get_name(soup):
    return soup.select_one('p.bikeNmae').get_text().strip()


def get_price(soup):
    div = soup.select_one('div.makerPriceRange')

    if not div or not div.select_one('div').get_text().strip().startswith('メーカー希望小売価格（税込）'):
        return None

    span = div.select_one('span.priceRange')

    if not span:
        return None

    price_string = span.get_text().strip()

    if '万' not in price_string or '円' not in price_string:
        return None

    price_1_string = price_string.split('万')[0]
    price_2_string = price_string.split('万')[1].split('円')[0]

    return (int(price_1_string) * 10_000 if price_1_string else 0) + (int(price_2_string) if price_2_string else 0)


def get_spec_value(soup, th_text, convert_fn):
    tr = first(filter(lambda tr: tr.select_one('th').get_text().strip() == th_text, soup.select('div#bike_model_info tr')))

    if not tr:
        return None

    return convert_fn(tr.select_one('td').get_text().strip())


def get_rows(category_url):
    for row_url in get_row_urls(category_url):
        sleep(10)  # サイトに迷惑をかけないよう、スリープして10秒待ちます。

        soup = get_soup(row_url)
        yield ((row_url, get_name(soup), get_price(soup)) +
               tuple(starmap(lambda caption, convert_fn: get_spec_value(soup, caption, convert_fn),
                             (('全長 (mm)', float),
                              ('全幅 (mm)', float),
                              ('全高 (mm)', float),
                              ('ホイールベース (mm)', float),
                              ('シート高 (mm)', float),
                              ('車両重量 (kg)', float),
                              ('気筒数', int),
                              ('シリンダ配列', str),
                              ('排気量 (cc)', float),
                              ('カム・バルブ駆動方式', str),
                              ('気筒あたりバルブ数', int),
                              ('最高出力（kW）', float),
                              ('最高出力回転数（rpm）', float),
                              ('最大トルク（N・m）', float),
                              ('最大トルク回転数（rpm）', float)))))


def main():
    # スクレイピングしてデータを取得します。
    rows = concat(map(lambda row: (0,) + row, get_rows('https://www.bikebros.co.jp/catalog/A01/')),  # スポーツ＆ツアラー
                  map(lambda row: (1,) + row, get_rows('https://www.bikebros.co.jp/catalog/B01/')),  # ネイキッド＆ストリート
                  map(lambda row: (2,) + row, get_rows('https://www.bikebros.co.jp/catalog/C01/')),  # オフロード＆モタード
                  map(lambda row: (3,) + row, get_rows('https://www.bikebros.co.jp/catalog/D01/')),  # アメリカン＆クルーザー
                  map(lambda row: (4,) + row, get_rows('https://www.bikebros.co.jp/catalog/E01/')),  # ビッグスクーター
                  map(lambda row: (5,) + row, get_rows('https://www.bikebros.co.jp/catalog/F01/')),  # 原付・スクーター
                  map(lambda row: (6,) + row, get_rows('https://www.bikebros.co.jp/catalog/G01/')))  # ビジネスバイク・ミニバイク

    # CSVとして出力します。
    with open('bike-bros-catalog.csv', 'w', newline="", encoding="UTF-8") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_NONNUMERIC)

        writer.writerow(('ジャンル',
                         'URL',
                         '車名',
                         '価格',
                         '全長 (mm)',
                         '全幅 (mm)',
                         '全高 (mm)',
                         'ホイールベース (mm)',
                         'シート高 (mm)',
                         '車両重量 (kg)',
                         '気筒数',
                         'シリンダ配列',
                         '排気量 (cc)',
                         'カム・バルブ駆動方式',
                         '気筒あたりバルブ数',
                         '最高出力（kW）',
                         '最高出力回転数（rpm）',
                         '最大トルク（N・m）',
                         '最大トルク回転数（rpm）'))

        for row in rows:
            # print(row)
            writer.writerow(row)


if __name__ == '__main__':
    main()
