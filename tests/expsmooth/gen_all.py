import os

# exit()


def generate_all():
    csvs="ausgdp.csv dji.csv gasprice.csv partx.csv utility.csv bonds.csv enplanements.csv hospital.csv ukcars.csv vehicles.csv canadagas.csv fmsales.csv jewelry.csv unemp.cci.csv visitors.csv carparts.csv freight.csv mcopper.csv usgdp.csv xrates.csv djiclose.csv frexport.csv msales.csv usnetelec.csv"
    
    
    for csv in csvs.split():
        name = csv.replace(".csv", "")
        filename = "tests/expsmooth/expsmooth_dataset_" + name + ".py";
        file = open(filename, "w");
        print("WRTITING_FILE" , filename);
        file.write("import tests.expsmooth.expsmooth_dataset_test as exps\n\n");
        file.write("exps.analyze_dataset(\"" + csv  + "\" , " +  str(2) + ");\n\n");
        file.write("exps.analyze_dataset(\"" + csv  + "\" , " +  str(4) + ");\n\n");
        file.write("exps.analyze_dataset(\"" + csv  + "\" , " +  str(8) + ");\n\n");
        file.write("exps.analyze_dataset(\"" + csv  + "\" , " +  str(12) + ");\n\n");
        file.close();


#generate_all();
