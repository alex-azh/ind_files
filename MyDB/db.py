import sqlite3
con = sqlite3.connect("mydb.db")

def CreateTable():
    cursor = con.cursor()
    q = "DROP TABLE IF EXISTS t_files; DROP INDEX if exists file_index_id; " \
        "CREATE TABLE t_files( " \
        "filename varchar primary key, " \
        "index_id int , " \
        "uptime timestamp DEFAULT CURRENT_TIMESTAMP);" \
        "CREATE INDEX file_index_id ON t_files (index_id);"
    cursor.executescript(q)
    con.commit()
    cursor.close()


def AddRow(filename, index):
    cursor = con.cursor()
    q = "insert into t_files(filename,index_id) values (?,?);"
    obj = (filename, index)
    cursor.execute(q, obj)
    con.commit()
    cursor.close()

def PrintTable():
    cursor = con.cursor()
    q = "select * from t_files;"
    cursor.execute(q)
    for res in cursor.fetchall():
        print(res)
    cursor.close()

CreateTable()
# AddRow("test",None)
# PrintTable()
