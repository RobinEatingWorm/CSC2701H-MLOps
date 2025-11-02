# Download data
mkdir -p data/raw
year=1996
for url in $(cat data/urls.txt); do
    curl -o "data/raw/${year}.csv" "${url}"
    ((year++))
done

# Preprocess data
python3 -m scripts.preprocessing
