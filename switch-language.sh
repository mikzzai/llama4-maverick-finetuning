#!/bin/bash

# Script to switch between Russian and English versions of the repository

if [ "$1" == "en" ]; then
    echo "Switching to English version..."
    
    # Backup current Russian files
    mv README.md README.md.ru
    mv GITHUB_PAGES_GUIDE.md GITHUB_PAGES_GUIDE.md.ru
    mv index.html index.html.ru
    
    # Rename English files to active files
    mv README.md.en README.md
    mv GITHUB_PAGES_GUIDE.md.en GITHUB_PAGES_GUIDE.md
    mv index.html.en index.html
    
    echo "Repository is now in English"
    
elif [ "$1" == "ru" ]; then
    echo "Switching to Russian version..."
    
    # Backup current English files
    mv README.md README.md.en
    mv GITHUB_PAGES_GUIDE.md GITHUB_PAGES_GUIDE.md.en
    mv index.html index.html.en
    
    # Rename Russian files to active files
    mv README.md.ru README.md
    mv GITHUB_PAGES_GUIDE.md.ru GITHUB_PAGES_GUIDE.md
    mv index.html.ru index.html
    
    echo "Repository is now in Russian"
    
else
    echo "Usage: $0 [en|ru]"
    echo "  en - Switch to English version"
    echo "  ru - Switch to Russian version"
    exit 1
fi

echo "Done!"
