import os
import shutil
import configparser
import smtplib
import os.path as path
import pandas as pd
import datetime as dt
import numpy as np
from pandas.io import gbq
from bigquery import get_client, errors
from logging import getLogger, StreamHandler, Formatter, basicConfig
from logging import INFO, DEBUG
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText


def init_logger(module_name, log_file_str):

    set_level = _common_conf['logger']['log_level']
    format_str = _common_conf['logger']['log_format']
    d_format_str = _common_conf['logger']['date_format']

    logger = getLogger(module_name)
    handler = StreamHandler()
    handler.setFormatter(Formatter(format_str))

    if set_level == 'INFO':
        log_level = INFO
    elif set_level == 'DEBUG':
        log_level = DEBUG
    else:
        log_level = INFO

    handler.setLevel(log_level)
    logger.setLevel(log_level)
    logger.addHandler(handler)

    par_dir = get_parent_dir_abspath()

    log_file_path_index = log_file_str.rfind('/')
    create_dir("{}/{}".format(par_dir, log_file_str[0:log_file_path_index]))
    log_file = log_file_str.format(dt.datetime.today().strftime("%Y%m%d"))

    # log dir create
    basicConfig(
        level=log_level,
        format=format_str,
        datefmt=d_format_str,
        filename="{}/{}".format(par_dir, log_file),
        filemode='a')

    return logger


def get_config(conf_name):
    """ コンフィグファイルの読み込み """
    config = configparser.ConfigParser()
    config.read(get_abspath(conf_name))

    return config


def get_common_config():
    """ 共通設定ファイルの取得 """
    return _common_conf


def get_parent_dir_abspath():
    """ 親ディレクトリの絶対パスの取得 """
    return path.abspath(os.path.dirname(__file__)) + path.sep + os.pardir


def get_abspath(file_name):
    """ 絶対パスの取得 """
    return "{}/{}".format(get_parent_dir_abspath(), file_name)


def create_dir(dir_path):
    """ 指定されたディレクトリが存在しなけば作成する """
    if not path.exists(dir_path):
        os.makedirs(dir_path)


def trans_str2datetime(str_date):
    """ transfer string to date
    Args: format is %Y-%m-%d
    """
    sp_nums = str_date.split('-')
    return dt.date(int(sp_nums[0]), int(sp_nums[1]), int(sp_nums[2]))


def diff_datetimes(start_str, end_str):
    """ 指定された文字列の期間の日数を返  """
    start_d = trans_str2datetime(start_str)
    end_d = trans_str2datetime(end_str)

    return (end_d - start_d).days


def send_error_msg(title, msg):
    """ エラー通知用関数 """
    print("[{}] {}".format(title, msg))

    mail_conf = _common_conf['mail']
    from_address = mail_conf['from_address']
    to_address = mail_conf['to_address']
    cc_address = mail_conf['cc_address']
    bcc_address = mail_conf['bcc_address']

    charset = "utf-8"
    mail = MIMEMultipart('alternative')
    mail["Subject"] = title
    mail["From"] = from_address
    mail["To"] = to_address

    if cc_address != "":
        mail["Cc"] = cc_address
    if bcc_address != "":
        mail["Bcc"] = bcc_address

    attachment_msg = MIMEText(msg, 'plain', charset)
    mail.attach(attachment_msg)

    smtp = smtplib.SMTP("localhost")
    smtp.send_message(msg=mail)
    smtp.close()


def get_list_conf_value(conf_value):
    if conf_value is None:
        return []
    else:
        return [x.strip() for x in conf_value.split(',')]


def get_dict_conf_value(conf_value):
    return eval("{" + conf_value + "}")


def insert_row(df, table, replace_val):
    if len(df.index) == 0:
        print("gbq insert records zero")
        return
    project_id = _common_conf['bigquery']['project_id']
    private_key_path = get_abspath(_common_conf['bigquery']['key_path'])
    dataset = _common_conf['bigquery']['dataset']

    # 10000ずつinset
    full_table = "{}.{}".format(dataset, table)
    client = get_client(
        json_key_file=private_key_path,
        readonly=False, swallow_results=False)

    if client.check_table(dataset, table):
            bq_limit = 10000
            q_num = len(df.index) // bq_limit
            for i in range(0, q_num + 1):
                client = get_client(
                    json_key_file=private_key_path,
                    readonly=False, swallow_results=False)
                ins_df = df[i * bq_limit: (i + 1) * bq_limit].replace(
                    np.nan, replace_val)

                row_dict = ins_df.to_dict(orient='index')
                row_data = [x for x in row_dict.values()]
                ret = client.push_rows(dataset, table, row_data)
                if 'insertErrors' in ret:
                    msg = "BigQuery Insert Error:\nsample:\n{}\nerror:\n{}"
                    raise Exception(msg.format(row_data[0:5], ret))
    else:
        # テーブルが存在しなければデータフレームをBigQueryに格納する
        print('{} CREATE TABLE'.format(full_table))
        gbq.to_gbq(df, full_table, project_id)


def backup_gbq(path,
               tmp_path, file_type, table_name, columns=None, replace_val=""):
    abs_path = get_abspath(path)
    create_dir(abs_path)
    files = os.listdir(abs_path)
    for f in files:
        if f.endswith(file_type):
            delimiter = ','
            read_path = "{}/{}".format(abs_path, f)
            print("bukcup file {} start".format(f))
            if file_type == 'tsv':
                delimiter = '\t'
                df = pd.read_csv(read_path,
                                 delimiter=delimiter,
                                 header=None, names=columns)
            else:
                df = pd.read_csv(read_path)
            insert_row(df, table_name, replace_val)
            tmp_abs_path = get_abspath(tmp_path)
            if not os.path.exists(tmp_abs_path):
                os.makedirs(tmp_abs_path)
            shutil.move(read_path, "{}/{}".format(tmp_abs_path, f))


def get_daily_start_end(start, end, table, date_col, media=None):
    """ start end が指定されない場合は、BQ最新保存日の次の日から三日前まで """
    print("{} - {}".format(start, end))
    limit_day_count = int(get_common_config()['bigquery']['daily_tmp_limit'])
    if start is None or end is None:
        tree_days_ago = dt.date.today() - dt.timedelta(limit_day_count)
        last_bq_date = get_last_bq_date(table, date_col, media)

        str_tree_days_ago = tree_days_ago.strftime("%Y-%m-%d")
        if last_bq_date is None:
            return (str_tree_days_ago, str_tree_days_ago)
        elif last_bq_date >= tree_days_ago:
            return (None, None)
        else:
            str_last_bq_date = last_bq_date.strftime("%Y-%m-%d")
            return (str_last_bq_date, str_tree_days_ago)
    return (start, end)


def get_last_bq_date(table, date_col, media=None, is_month=False):
    private_key_path = get_abspath(_common_conf['bigquery']['key_path'])
    dataset = _common_conf['bigquery']['dataset']

    client = get_client(json_key_file=private_key_path, readonly=False)
    if not client.check_table(dataset, table):
        return None

    full_table = "{}.{}".format(dataset, table)

    if media is None:
        query = """
            SELECT
              {0} as last_date
            FROM {1}
            ORDER BY
              {0} DESC
            LIMIT 1
        """.format(date_col, full_table)
    else:
        query = """
          SELECT
            {0} as last_date
          FROM {1}
          WHERE
            Media = '{2}'
          ORDER BY
            {0} DESC
          LIMIT 1
        """.format(
            date_col, full_table, media)
    try:
        job_id, results = client.query(query, timeout=80)
    except errors.BigQueryTimeoutException as e:
        raise e
    if len(results) > 0:
        ld = results[0]['last_date']
        if isinstance(ld, str):
            strs = ld.split('-')
            if is_month:
                return dt.date(int(strs[0]), int(strs[1]), 1)
            else:
                return dt.date(int(strs[0]), int(strs[1]), int(strs[2]))
        elif isinstance(ld, dt.date):
            return ld
    else:
        return None


_common_conf = get_config("common.ini")
