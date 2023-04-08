import pytsk3

# Открываем образ диска NTFS
img = pytsk3.Img_Info('disk.dd')

# Открываем раздел NTFS
fs = pytsk3.FS_Info(img, offset=0)

# Получаем объект MFT
mft = fs.open_meta(0)

# Проходимся по всем записям MFT
for record in mft:
    # Проверяем, был ли файл удален
    if record.info.meta.type == pytsk3.TSK_FS_META_FLAG_UNALLOC:
        # Открываем файл для чтения
        file_object = fs.open_meta(record.info.meta.addr)
        data = file_object.read_random(0, record.info.meta.size)
        print(f"Файл {record.info.name.name} был удален, данные: {data}")

        # Закрываем файл
        file_object.close()

# Закрываем объект MFT и раздел NTFS
mft.close()
fs.close()
img.close()
