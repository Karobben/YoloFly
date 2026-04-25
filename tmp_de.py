import os

List = os.listdir("csv")

List=[os.path.join("csv", i) for i in List]

List[0]

# Get the size of the file in bytes
Empty = []
for i in List:
    size = os.path.getsize(i)
    if size == 0:
        Empty.append(i)

len(Empty)

# Remove empty files
for i in Empty:
    os.remove(i)

