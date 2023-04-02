import psycopg2

conn = psycopg2.connect(
    host="localhost",
    database="testdb",
    user="postgres",
    password="postgres"
)


def executeQuery(query, args):
    """
    Выполнить SQL-запрос.
    :param query: SQL-запрос в виде строки
    :param args: параметры [...]
    :return:
    """
    try:
        with conn.cursor() as cur:
            cur.execute(query, args)
        return True, None
    except Exception as e:
        return False, e


def returnQuery(query, args):
    with conn.cursor() as cur:
        cur.execute(query, args)
        r = cur.fetchall()
        columns = list(map(lambda x: x[0], cur.description))
        r = list(map(lambda x: dict(zip(columns, x)), r))


def CreateTableAndDropIfExists():
    with conn.cursor() as cur:
        cur.execute("drop table if exists t_files; "
                    "CREATE TABLE t_files ("
                    "filepath text NOT NULL primary key, "
                    "embd FLOAT[], "
                    "index_id INTEGER,"
                    "whentime timestamp default now()); "
                    "drop table if exists t_files_notadded; "
                    "CREATE TABLE t_files_notadded ("
                    "filepath text not null primary key, "
                    "whentime timestamp default now());")


def InsertNewFile(filepath, index_id=None, embd=None):
    if index_id == None:
        return executeQuery("insert into t_files_notadded(filepath) values(%s);", [filepath])
    return executeQuery("insert into t_files(filepath,index_id,embd) values(%s,%s,%s);", [filepath, index_id, embd])
