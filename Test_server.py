import pymongo
import certifi

# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    client = pymongo.MongoClient("mongodb+srv://DKD:24011995@cluster0.jxdnt.mongodb.net/myFirstDatabase?retryWrites=true&w=majority",tlsCAFile=certifi.where())
    #tlsCAFile=certifi.where()
    db = client['test']
    mycol = db['test1']
    #print(client.list_database_names())

    #dic = {'name': 'KhanhDung', 'age': 26}
    #x = col.insert_one(dic)
    #print(x)

    myquery = { "name": "KhanhDung" }
    newvalues = { "$set": { "name": "KhanhDungDao" } }

    mycol.update_one(myquery, newvalues)

    #print "customers" after the update:
    for x in mycol.find():
        print(x)
