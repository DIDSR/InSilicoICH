cd sphinx

make html

mv build/html/index.html build/html/docs.html

cp -r build/html/* ../*