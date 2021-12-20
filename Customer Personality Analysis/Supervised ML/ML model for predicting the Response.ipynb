{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "kernelspec": {
      "display_name": "Python 3",
      "language": "python",
      "name": "python3"
    },
    "language_info": {
      "codemirror_mode": {
        "name": "ipython",
        "version": 3
      },
      "file_extension": ".py",
      "mimetype": "text/x-python",
      "name": "python",
      "nbconvert_exporter": "python",
      "pygments_lexer": "ipython3",
      "version": "3.7.6"
    },
    "colab": {
      "name": "final model",
      "provenance": [],
      "collapsed_sections": [],
      "include_colab_link": true
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/Shakibap/Data-Science-Portfo/blob/Data-Science/final_model.ipynb\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "U-dg6_c-Q9ke"
      },
      "source": [
        "# **Predicting the Response**"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "EU6fp1uOM73V"
      },
      "source": [
        "import pandas as pd\n",
        "import numpy as np\n",
        "import seaborn as sns\n",
        "import matplotlib.pyplot as plt\n",
        "%matplotlib inline\n",
        "from sklearn.impute import SimpleImputer\n",
        "from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier\n",
        "from sklearn.ensemble import GradientBoostingClassifier\n",
        "from xgboost import XGBClassifier\n",
        "from sklearn.svm import SVC\n",
        "from sklearn.neighbors import KNeighborsClassifier\n",
        "from sklearn.decomposition import PCA\n",
        "from sklearn.preprocessing import MinMaxScaler\n",
        "from sklearn.metrics import classification_report \n",
        "from sklearn.model_selection import train_test_split, RandomizedSearchCV\n",
        "from sklearn.feature_selection import mutual_info_classif"
      ],
      "execution_count": 2,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "LViMbb53NPM8"
      },
      "source": [
        "from google.colab import files\n",
        "#files.upload()"
      ],
      "execution_count": 99,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "9LPqR90XM73h"
      },
      "source": [
        "df = pd.read_csv('marketing_campaign.csv', sep='\\t')"
      ],
      "execution_count": 4,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 226
        },
        "id": "3Y-hHtutM73h",
        "outputId": "6ee8bac9-674c-4e1f-d360-96374d51ff5f"
      },
      "source": [
        "df.head()"
      ],
      "execution_count": 96,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/html": [
              "<div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>ID</th>\n",
              "      <th>Year_Birth</th>\n",
              "      <th>Education</th>\n",
              "      <th>Marital_Status</th>\n",
              "      <th>Income</th>\n",
              "      <th>Kidhome</th>\n",
              "      <th>Teenhome</th>\n",
              "      <th>Dt_Customer</th>\n",
              "      <th>Recency</th>\n",
              "      <th>MntWines</th>\n",
              "      <th>MntFruits</th>\n",
              "      <th>MntMeatProducts</th>\n",
              "      <th>MntFishProducts</th>\n",
              "      <th>MntSweetProducts</th>\n",
              "      <th>MntGoldProds</th>\n",
              "      <th>NumDealsPurchases</th>\n",
              "      <th>NumWebPurchases</th>\n",
              "      <th>NumCatalogPurchases</th>\n",
              "      <th>NumStorePurchases</th>\n",
              "      <th>NumWebVisitsMonth</th>\n",
              "      <th>AcceptedCmp3</th>\n",
              "      <th>AcceptedCmp4</th>\n",
              "      <th>AcceptedCmp5</th>\n",
              "      <th>AcceptedCmp1</th>\n",
              "      <th>AcceptedCmp2</th>\n",
              "      <th>Complain</th>\n",
              "      <th>Z_CostContact</th>\n",
              "      <th>Z_Revenue</th>\n",
              "      <th>Response</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>0</th>\n",
              "      <td>5524</td>\n",
              "      <td>1957</td>\n",
              "      <td>Graduation</td>\n",
              "      <td>Single</td>\n",
              "      <td>58138.0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>2012-04-09</td>\n",
              "      <td>58</td>\n",
              "      <td>635</td>\n",
              "      <td>88</td>\n",
              "      <td>546</td>\n",
              "      <td>172</td>\n",
              "      <td>88</td>\n",
              "      <td>88</td>\n",
              "      <td>3</td>\n",
              "      <td>8</td>\n",
              "      <td>10</td>\n",
              "      <td>4</td>\n",
              "      <td>7</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>3</td>\n",
              "      <td>11</td>\n",
              "      <td>1</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>1</th>\n",
              "      <td>2174</td>\n",
              "      <td>1954</td>\n",
              "      <td>Graduation</td>\n",
              "      <td>Single</td>\n",
              "      <td>46344.0</td>\n",
              "      <td>1</td>\n",
              "      <td>1</td>\n",
              "      <td>2014-08-03</td>\n",
              "      <td>38</td>\n",
              "      <td>11</td>\n",
              "      <td>1</td>\n",
              "      <td>6</td>\n",
              "      <td>2</td>\n",
              "      <td>1</td>\n",
              "      <td>6</td>\n",
              "      <td>2</td>\n",
              "      <td>1</td>\n",
              "      <td>1</td>\n",
              "      <td>2</td>\n",
              "      <td>5</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>3</td>\n",
              "      <td>11</td>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>2</th>\n",
              "      <td>4141</td>\n",
              "      <td>1965</td>\n",
              "      <td>Graduation</td>\n",
              "      <td>Couple</td>\n",
              "      <td>71613.0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>2013-08-21</td>\n",
              "      <td>26</td>\n",
              "      <td>426</td>\n",
              "      <td>49</td>\n",
              "      <td>127</td>\n",
              "      <td>111</td>\n",
              "      <td>21</td>\n",
              "      <td>42</td>\n",
              "      <td>1</td>\n",
              "      <td>8</td>\n",
              "      <td>2</td>\n",
              "      <td>10</td>\n",
              "      <td>4</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>3</td>\n",
              "      <td>11</td>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>3</th>\n",
              "      <td>6182</td>\n",
              "      <td>1984</td>\n",
              "      <td>Graduation</td>\n",
              "      <td>Couple</td>\n",
              "      <td>26646.0</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>2014-10-02</td>\n",
              "      <td>26</td>\n",
              "      <td>11</td>\n",
              "      <td>4</td>\n",
              "      <td>20</td>\n",
              "      <td>10</td>\n",
              "      <td>3</td>\n",
              "      <td>5</td>\n",
              "      <td>2</td>\n",
              "      <td>2</td>\n",
              "      <td>0</td>\n",
              "      <td>4</td>\n",
              "      <td>6</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>3</td>\n",
              "      <td>11</td>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>4</th>\n",
              "      <td>5324</td>\n",
              "      <td>1981</td>\n",
              "      <td>PhD</td>\n",
              "      <td>Couple</td>\n",
              "      <td>58293.0</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>2014-01-19</td>\n",
              "      <td>94</td>\n",
              "      <td>173</td>\n",
              "      <td>43</td>\n",
              "      <td>118</td>\n",
              "      <td>46</td>\n",
              "      <td>27</td>\n",
              "      <td>15</td>\n",
              "      <td>5</td>\n",
              "      <td>5</td>\n",
              "      <td>3</td>\n",
              "      <td>6</td>\n",
              "      <td>5</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>3</td>\n",
              "      <td>11</td>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "</div>"
            ],
            "text/plain": [
              "     ID  Year_Birth   Education  ... Z_CostContact  Z_Revenue  Response\n",
              "0  5524        1957  Graduation  ...             3         11         1\n",
              "1  2174        1954  Graduation  ...             3         11         0\n",
              "2  4141        1965  Graduation  ...             3         11         0\n",
              "3  6182        1984  Graduation  ...             3         11         0\n",
              "4  5324        1981         PhD  ...             3         11         0\n",
              "\n",
              "[5 rows x 29 columns]"
            ]
          },
          "metadata": {},
          "execution_count": 96
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "0FdmL40LRpm7"
      },
      "source": [
        "# **Data Cleaning**"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "P-66l8UpXh_9",
        "outputId": "ff1cc2fe-37be-4195-c9d7-cc23ce9313f3"
      },
      "source": [
        "df.shape"
      ],
      "execution_count": 98,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "(2232, 29)"
            ]
          },
          "metadata": {},
          "execution_count": 98
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "5Nt-7gSOM73n",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "ce1810a3-a343-42e8-fe9f-f26be103319d"
      },
      "source": [
        "df.info()"
      ],
      "execution_count": 97,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "<class 'pandas.core.frame.DataFrame'>\n",
            "Int64Index: 2232 entries, 0 to 2239\n",
            "Data columns (total 29 columns):\n",
            " #   Column               Non-Null Count  Dtype         \n",
            "---  ------               --------------  -----         \n",
            " 0   ID                   2232 non-null   int64         \n",
            " 1   Year_Birth           2232 non-null   int64         \n",
            " 2   Education            2232 non-null   object        \n",
            " 3   Marital_Status       2232 non-null   object        \n",
            " 4   Income               2208 non-null   float64       \n",
            " 5   Kidhome              2232 non-null   int64         \n",
            " 6   Teenhome             2232 non-null   int64         \n",
            " 7   Dt_Customer          2232 non-null   datetime64[ns]\n",
            " 8   Recency              2232 non-null   int64         \n",
            " 9   MntWines             2232 non-null   int64         \n",
            " 10  MntFruits            2232 non-null   int64         \n",
            " 11  MntMeatProducts      2232 non-null   int64         \n",
            " 12  MntFishProducts      2232 non-null   int64         \n",
            " 13  MntSweetProducts     2232 non-null   int64         \n",
            " 14  MntGoldProds         2232 non-null   int64         \n",
            " 15  NumDealsPurchases    2232 non-null   int64         \n",
            " 16  NumWebPurchases      2232 non-null   int64         \n",
            " 17  NumCatalogPurchases  2232 non-null   int64         \n",
            " 18  NumStorePurchases    2232 non-null   int64         \n",
            " 19  NumWebVisitsMonth    2232 non-null   int64         \n",
            " 20  AcceptedCmp3         2232 non-null   int64         \n",
            " 21  AcceptedCmp4         2232 non-null   int64         \n",
            " 22  AcceptedCmp5         2232 non-null   int64         \n",
            " 23  AcceptedCmp1         2232 non-null   int64         \n",
            " 24  AcceptedCmp2         2232 non-null   int64         \n",
            " 25  Complain             2232 non-null   int64         \n",
            " 26  Z_CostContact        2232 non-null   int64         \n",
            " 27  Z_Revenue            2232 non-null   int64         \n",
            " 28  Response             2232 non-null   int64         \n",
            "dtypes: datetime64[ns](1), float64(1), int64(25), object(2)\n",
            "memory usage: 523.1+ KB\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "YUBngZyEM73q"
      },
      "source": [
        "df.Dt_Customer = pd.to_datetime(df.Dt_Customer)"
      ],
      "execution_count": 7,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "-uUzchK2OxPK"
      },
      "source": [
        "**finding the outliers**"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 301
        },
        "id": "XOSizE5KM73t",
        "outputId": "12a07e9e-db39-4bf0-8a6e-6e7f21dfa6b2"
      },
      "source": [
        "sns.set_context(\"poster\")\n",
        "sns.scatterplot(x=df.index, y= df.Income)"
      ],
      "execution_count": 8,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "<matplotlib.axes._subplots.AxesSubplot at 0x7f76110eb890>"
            ]
          },
          "metadata": {},
          "execution_count": 8
        },
        {
          "output_type": "display_data",
          "data": {
            "image/png": "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\n",
            "text/plain": [
              "<Figure size 432x288 with 1 Axes>"
            ]
          },
          "metadata": {
            "needs_background": "light"
          }
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "NMAOPj_TM73u",
        "outputId": "bb669b85-e342-4edd-e4a9-09cf0c9692ca"
      },
      "source": [
        "df.Income.idxmax()"
      ],
      "execution_count": 9,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "2233"
            ]
          },
          "metadata": {},
          "execution_count": 9
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "c8tK7HHmM73u"
      },
      "source": [
        "#removing the outlier data point\n",
        "df.drop(index=2233, inplace=True)"
      ],
      "execution_count": 10,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 301
        },
        "id": "3H83LQhPM73w",
        "outputId": "46a9d87b-0f4f-4faf-b89a-784b04f45fda"
      },
      "source": [
        "sns.scatterplot(x=df.index, y= df.Income)"
      ],
      "execution_count": 11,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "<matplotlib.axes._subplots.AxesSubplot at 0x7f75f0bbd890>"
            ]
          },
          "metadata": {},
          "execution_count": 11
        },
        {
          "output_type": "display_data",
          "data": {
            "image/png": "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\n",
            "text/plain": [
              "<Figure size 432x288 with 1 Axes>"
            ]
          },
          "metadata": {
            "needs_background": "light"
          }
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 330
        },
        "id": "u9aGu8IpM73x",
        "outputId": "4028d09c-fc43-4996-8f6c-da358dceaf74"
      },
      "source": [
        "sns.scatterplot(x=df.Year_Birth, y=df.index)"
      ],
      "execution_count": 12,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "<matplotlib.axes._subplots.AxesSubplot at 0x7f75f06f8410>"
            ]
          },
          "metadata": {},
          "execution_count": 12
        },
        {
          "output_type": "display_data",
          "data": {
            "image/png": "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\n",
            "text/plain": [
              "<Figure size 432x288 with 1 Axes>"
            ]
          },
          "metadata": {
            "needs_background": "light"
          }
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 163
        },
        "id": "COuOflZXM73y",
        "outputId": "4dce0c35-73b8-4ba4-9dd8-6e2a5d188f7e"
      },
      "source": [
        "df[df[\"Year_Birth\"] < 1915]"
      ],
      "execution_count": 13,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/html": [
              "<div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>ID</th>\n",
              "      <th>Year_Birth</th>\n",
              "      <th>Education</th>\n",
              "      <th>Marital_Status</th>\n",
              "      <th>Income</th>\n",
              "      <th>Kidhome</th>\n",
              "      <th>Teenhome</th>\n",
              "      <th>Dt_Customer</th>\n",
              "      <th>Recency</th>\n",
              "      <th>MntWines</th>\n",
              "      <th>MntFruits</th>\n",
              "      <th>MntMeatProducts</th>\n",
              "      <th>MntFishProducts</th>\n",
              "      <th>MntSweetProducts</th>\n",
              "      <th>MntGoldProds</th>\n",
              "      <th>NumDealsPurchases</th>\n",
              "      <th>NumWebPurchases</th>\n",
              "      <th>NumCatalogPurchases</th>\n",
              "      <th>NumStorePurchases</th>\n",
              "      <th>NumWebVisitsMonth</th>\n",
              "      <th>AcceptedCmp3</th>\n",
              "      <th>AcceptedCmp4</th>\n",
              "      <th>AcceptedCmp5</th>\n",
              "      <th>AcceptedCmp1</th>\n",
              "      <th>AcceptedCmp2</th>\n",
              "      <th>Complain</th>\n",
              "      <th>Z_CostContact</th>\n",
              "      <th>Z_Revenue</th>\n",
              "      <th>Response</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>192</th>\n",
              "      <td>7829</td>\n",
              "      <td>1900</td>\n",
              "      <td>2n Cycle</td>\n",
              "      <td>Divorced</td>\n",
              "      <td>36640.0</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>2013-09-26</td>\n",
              "      <td>99</td>\n",
              "      <td>15</td>\n",
              "      <td>6</td>\n",
              "      <td>8</td>\n",
              "      <td>7</td>\n",
              "      <td>4</td>\n",
              "      <td>25</td>\n",
              "      <td>1</td>\n",
              "      <td>2</td>\n",
              "      <td>1</td>\n",
              "      <td>2</td>\n",
              "      <td>5</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>1</td>\n",
              "      <td>3</td>\n",
              "      <td>11</td>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>239</th>\n",
              "      <td>11004</td>\n",
              "      <td>1893</td>\n",
              "      <td>2n Cycle</td>\n",
              "      <td>Single</td>\n",
              "      <td>60182.0</td>\n",
              "      <td>0</td>\n",
              "      <td>1</td>\n",
              "      <td>2014-05-17</td>\n",
              "      <td>23</td>\n",
              "      <td>8</td>\n",
              "      <td>0</td>\n",
              "      <td>5</td>\n",
              "      <td>7</td>\n",
              "      <td>0</td>\n",
              "      <td>2</td>\n",
              "      <td>1</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>2</td>\n",
              "      <td>4</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>3</td>\n",
              "      <td>11</td>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>339</th>\n",
              "      <td>1150</td>\n",
              "      <td>1899</td>\n",
              "      <td>PhD</td>\n",
              "      <td>Together</td>\n",
              "      <td>83532.0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>2013-09-26</td>\n",
              "      <td>36</td>\n",
              "      <td>755</td>\n",
              "      <td>144</td>\n",
              "      <td>562</td>\n",
              "      <td>104</td>\n",
              "      <td>64</td>\n",
              "      <td>224</td>\n",
              "      <td>1</td>\n",
              "      <td>4</td>\n",
              "      <td>6</td>\n",
              "      <td>4</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>3</td>\n",
              "      <td>11</td>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "</div>"
            ],
            "text/plain": [
              "        ID  Year_Birth Education  ... Z_CostContact  Z_Revenue  Response\n",
              "192   7829        1900  2n Cycle  ...             3         11         0\n",
              "239  11004        1893  2n Cycle  ...             3         11         0\n",
              "339   1150        1899       PhD  ...             3         11         0\n",
              "\n",
              "[3 rows x 29 columns]"
            ]
          },
          "metadata": {},
          "execution_count": 13
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "BW2sWVGmM73z"
      },
      "source": [
        "\n",
        "#removing the outlier data point\n",
        "df.drop(index=[192, 239, 339] , inplace=True)"
      ],
      "execution_count": 14,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "_YMazA8bPasx"
      },
      "source": [
        "**concatenating the similar features**"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "A_rV9-5RM73z",
        "outputId": "f687daaa-57af-4de6-c0e2-98b51f3f45b4"
      },
      "source": [
        "df.Marital_Status.unique()"
      ],
      "execution_count": 15,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "array(['Single', 'Together', 'Married', 'Divorced', 'Widow', 'Alone',\n",
              "       'Absurd', 'YOLO'], dtype=object)"
            ]
          },
          "metadata": {},
          "execution_count": 15
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "74yCkz0gM730"
      },
      "source": [
        "*Divorced, Widow and Alone are considered as Single*\n",
        "\n",
        "*Together and Married are considered as Couple* "
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "Odv57MI4M733"
      },
      "source": [
        "#changing Divorced, Widow and Alone data points to Single. Also considering Together and Married data points as Couple.\n",
        "df.Marital_Status.replace('Divorced' , 'Single', inplace=True)\n",
        "df.Marital_Status.replace('Widow' , 'Single', inplace=True)\n",
        "df.Marital_Status.replace('Alone' , 'Single', inplace=True)\n",
        "df.Marital_Status.replace('Married' , 'Couple', inplace=True)\n",
        "df.Marital_Status.replace('Together' , 'Couple', inplace=True)"
      ],
      "execution_count": 16,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "voI8jCvJM734",
        "outputId": "1d66f864-d2f2-4999-bf30-98e32a47c0f2"
      },
      "source": [
        "df.Marital_Status.unique()"
      ],
      "execution_count": 17,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "array(['Single', 'Couple', 'Absurd', 'YOLO'], dtype=object)"
            ]
          },
          "metadata": {},
          "execution_count": 17
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 195
        },
        "id": "BM3mJZkjM735",
        "outputId": "53369de0-c59b-4ceb-9610-96afe933d0c0"
      },
      "source": [
        "df[df.Marital_Status.isin(['Absurd', 'YOLO'])]"
      ],
      "execution_count": 18,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/html": [
              "<div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>ID</th>\n",
              "      <th>Year_Birth</th>\n",
              "      <th>Education</th>\n",
              "      <th>Marital_Status</th>\n",
              "      <th>Income</th>\n",
              "      <th>Kidhome</th>\n",
              "      <th>Teenhome</th>\n",
              "      <th>Dt_Customer</th>\n",
              "      <th>Recency</th>\n",
              "      <th>MntWines</th>\n",
              "      <th>MntFruits</th>\n",
              "      <th>MntMeatProducts</th>\n",
              "      <th>MntFishProducts</th>\n",
              "      <th>MntSweetProducts</th>\n",
              "      <th>MntGoldProds</th>\n",
              "      <th>NumDealsPurchases</th>\n",
              "      <th>NumWebPurchases</th>\n",
              "      <th>NumCatalogPurchases</th>\n",
              "      <th>NumStorePurchases</th>\n",
              "      <th>NumWebVisitsMonth</th>\n",
              "      <th>AcceptedCmp3</th>\n",
              "      <th>AcceptedCmp4</th>\n",
              "      <th>AcceptedCmp5</th>\n",
              "      <th>AcceptedCmp1</th>\n",
              "      <th>AcceptedCmp2</th>\n",
              "      <th>Complain</th>\n",
              "      <th>Z_CostContact</th>\n",
              "      <th>Z_Revenue</th>\n",
              "      <th>Response</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>2093</th>\n",
              "      <td>7734</td>\n",
              "      <td>1993</td>\n",
              "      <td>Graduation</td>\n",
              "      <td>Absurd</td>\n",
              "      <td>79244.0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>2012-12-19</td>\n",
              "      <td>58</td>\n",
              "      <td>471</td>\n",
              "      <td>102</td>\n",
              "      <td>125</td>\n",
              "      <td>212</td>\n",
              "      <td>61</td>\n",
              "      <td>245</td>\n",
              "      <td>1</td>\n",
              "      <td>4</td>\n",
              "      <td>10</td>\n",
              "      <td>7</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>1</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>3</td>\n",
              "      <td>11</td>\n",
              "      <td>1</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>2134</th>\n",
              "      <td>4369</td>\n",
              "      <td>1957</td>\n",
              "      <td>Master</td>\n",
              "      <td>Absurd</td>\n",
              "      <td>65487.0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>2014-10-01</td>\n",
              "      <td>48</td>\n",
              "      <td>240</td>\n",
              "      <td>67</td>\n",
              "      <td>500</td>\n",
              "      <td>199</td>\n",
              "      <td>0</td>\n",
              "      <td>163</td>\n",
              "      <td>3</td>\n",
              "      <td>3</td>\n",
              "      <td>5</td>\n",
              "      <td>6</td>\n",
              "      <td>2</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>3</td>\n",
              "      <td>11</td>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>2177</th>\n",
              "      <td>492</td>\n",
              "      <td>1973</td>\n",
              "      <td>PhD</td>\n",
              "      <td>YOLO</td>\n",
              "      <td>48432.0</td>\n",
              "      <td>0</td>\n",
              "      <td>1</td>\n",
              "      <td>2012-10-18</td>\n",
              "      <td>3</td>\n",
              "      <td>322</td>\n",
              "      <td>3</td>\n",
              "      <td>50</td>\n",
              "      <td>4</td>\n",
              "      <td>3</td>\n",
              "      <td>42</td>\n",
              "      <td>5</td>\n",
              "      <td>7</td>\n",
              "      <td>1</td>\n",
              "      <td>6</td>\n",
              "      <td>8</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>3</td>\n",
              "      <td>11</td>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>2202</th>\n",
              "      <td>11133</td>\n",
              "      <td>1973</td>\n",
              "      <td>PhD</td>\n",
              "      <td>YOLO</td>\n",
              "      <td>48432.0</td>\n",
              "      <td>0</td>\n",
              "      <td>1</td>\n",
              "      <td>2012-10-18</td>\n",
              "      <td>3</td>\n",
              "      <td>322</td>\n",
              "      <td>3</td>\n",
              "      <td>50</td>\n",
              "      <td>4</td>\n",
              "      <td>3</td>\n",
              "      <td>42</td>\n",
              "      <td>5</td>\n",
              "      <td>7</td>\n",
              "      <td>1</td>\n",
              "      <td>6</td>\n",
              "      <td>8</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>3</td>\n",
              "      <td>11</td>\n",
              "      <td>1</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "</div>"
            ],
            "text/plain": [
              "         ID  Year_Birth   Education  ... Z_CostContact  Z_Revenue  Response\n",
              "2093   7734        1993  Graduation  ...             3         11         1\n",
              "2134   4369        1957      Master  ...             3         11         0\n",
              "2177    492        1973         PhD  ...             3         11         0\n",
              "2202  11133        1973         PhD  ...             3         11         1\n",
              "\n",
              "[4 rows x 29 columns]"
            ]
          },
          "metadata": {},
          "execution_count": 18
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "bamP_L73M739"
      },
      "source": [
        "#removing outliers\n",
        "df.drop(index=[2093, 2134, 2177, 2202] , inplace=True)"
      ],
      "execution_count": 19,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "wWd011M9M74O"
      },
      "source": [
        "# **Data Preprocessing**"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "9RENXcXTM74P"
      },
      "source": [
        "X_full = df.loc[ : , 'Year_Birth': 'Z_Revenue']\n",
        "y_full = df.Response"
      ],
      "execution_count": 20,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "defsxEdLM74P"
      },
      "source": [
        "X_full['Dt_Customer'] = X_full.Dt_Customer.astype('int64')"
      ],
      "execution_count": 21,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "d-aU7DWPM74Q"
      },
      "source": [
        "X_train, X_test, y_train, y_test = train_test_split(X_full, y_full, random_state=1)"
      ],
      "execution_count": 22,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "XKoRP0B8XZRB"
      },
      "source": [
        "#X_train.isnull().sum()\n",
        "#X_test.isnull().sum()"
      ],
      "execution_count": 23,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "Fzcw_zHXM73v"
      },
      "source": [
        "#imputing missing values\n",
        "imputer = SimpleImputer()\n",
        "\n",
        "x_train_income = imputer.fit_transform(X_train[['Income']])\n",
        "x_test_income = imputer.transform(X_test[['Income']])\n",
        "\n",
        "X_train.Income = x_train_income\n",
        "X_test.Income = x_test_income"
      ],
      "execution_count": 24,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "dY7kQzKdM74Q"
      },
      "source": [
        "#encoding categorical variables\n",
        "X_train = pd.get_dummies(X_train)\n",
        "X_test = pd.get_dummies(X_test)"
      ],
      "execution_count": 25,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "d_wYjVYfM74Q",
        "outputId": "fe6ba27b-36b4-41ad-e763-e79af8894f5a"
      },
      "source": [
        "#finding the percent of 1 in label\n",
        "y_full.sum()/y_full.shape[0]"
      ],
      "execution_count": 26,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "0.14874551971326164"
            ]
          },
          "metadata": {},
          "execution_count": 26
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "wE9oczrDab1m"
      },
      "source": [
        "*the labels are in two categories, 0 and 1. because the number of 1 is in minority (about 15%), the **f1_score** of label 1 is the **evaluation metric**.*"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "TWGcbyCvbru0"
      },
      "source": [
        "# **Modelling**"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "hXK79YY5E7gS"
      },
      "source": [
        "# *choosing potential classifiers*"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "hLIdxQ63b-jX"
      },
      "source": [
        "Random Forest Classifier\n"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "ssQpE1WrM74R",
        "outputId": "9d5e1511-ea6c-4434-99cb-51860261b9dc"
      },
      "source": [
        "#fitting data to the model\n",
        "model = RandomForestClassifier()\n",
        "model.fit(X_train, y_train)\n",
        "predictions = model.predict(X_test)\n",
        "print(classification_report(y_test, predictions))"
      ],
      "execution_count": 27,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "              precision    recall  f1-score   support\n",
            "\n",
            "           0       0.89      0.98      0.93       470\n",
            "           1       0.74      0.35      0.48        88\n",
            "\n",
            "    accuracy                           0.88       558\n",
            "   macro avg       0.81      0.66      0.70       558\n",
            "weighted avg       0.87      0.88      0.86       558\n",
            "\n"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "cIs73aeic5wv"
      },
      "source": [
        "AdaBoost Classifier"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "n6BpbAC1M74V",
        "outputId": "be4b5ece-05af-4256-9132-ad73f43aefbb"
      },
      "source": [
        "model = AdaBoostClassifier()\n",
        "model.fit(X_train, y_train)\n",
        "predictions = model.predict(X_test)\n",
        "print(classification_report(y_test, predictions))"
      ],
      "execution_count": 28,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "              precision    recall  f1-score   support\n",
            "\n",
            "           0       0.90      0.96      0.93       470\n",
            "           1       0.67      0.45      0.54        88\n",
            "\n",
            "    accuracy                           0.88       558\n",
            "   macro avg       0.79      0.71      0.74       558\n",
            "weighted avg       0.87      0.88      0.87       558\n",
            "\n"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "ud_h28PQe4P5"
      },
      "source": [
        "Gradient Boosting Classifier"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "LMV1B01OeNOv",
        "outputId": "fae550fb-ff48-42ed-ad09-16412ae151b0"
      },
      "source": [
        "model = GradientBoostingClassifier()\n",
        "model.fit(X_train, y_train)\n",
        "predictions = model.predict(X_test)\n",
        "print(classification_report(y_test, predictions))"
      ],
      "execution_count": 29,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "              precision    recall  f1-score   support\n",
            "\n",
            "           0       0.90      0.97      0.93       470\n",
            "           1       0.70      0.42      0.52        88\n",
            "\n",
            "    accuracy                           0.88       558\n",
            "   macro avg       0.80      0.69      0.73       558\n",
            "weighted avg       0.87      0.88      0.87       558\n",
            "\n"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "_49ZshZXcWT0"
      },
      "source": [
        "XGB Classifier"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "zvWVwLQfM74R",
        "outputId": "28f73f0a-7a6a-40bc-9b6e-d9ada654f75a"
      },
      "source": [
        "model = XGBClassifier()\n",
        "model.fit(X_train, y_train)\n",
        "predictions = model.predict(X_test)\n",
        "print(classification_report(y_test, predictions))"
      ],
      "execution_count": 30,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "              precision    recall  f1-score   support\n",
            "\n",
            "           0       0.90      0.97      0.93       470\n",
            "           1       0.70      0.43      0.54        88\n",
            "\n",
            "    accuracy                           0.88       558\n",
            "   macro avg       0.80      0.70      0.73       558\n",
            "weighted avg       0.87      0.88      0.87       558\n",
            "\n"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "NJYIhzZGfsS2"
      },
      "source": [
        "KNeighbors Classifier"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "qsNQVfW3M74Y"
      },
      "source": [
        "scaler = MinMaxScaler()\n",
        "\n",
        "X_train_scaled = X_train.copy()\n",
        "X_test_scaled = X_test.copy()\n",
        "\n",
        "X_train_scaled.loc[:] = scaler.fit_transform(X_train)\n",
        "X_test_scaled.loc[:] = scaler.transform(X_test)"
      ],
      "execution_count": 31,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "le1r1y65M74Z",
        "outputId": "f5d47eef-99d7-4e6e-c43f-2cfa321a9a04"
      },
      "source": [
        "model = KNeighborsClassifier()\n",
        "model.fit(X_train_scaled, y_train)\n",
        "predictions = model.predict(X_test_scaled)\n",
        "print(classification_report(y_test, predictions))"
      ],
      "execution_count": 32,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "              precision    recall  f1-score   support\n",
            "\n",
            "           0       0.87      0.96      0.91       470\n",
            "           1       0.52      0.25      0.34        88\n",
            "\n",
            "    accuracy                           0.85       558\n",
            "   macro avg       0.70      0.60      0.63       558\n",
            "weighted avg       0.82      0.85      0.82       558\n",
            "\n"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "r7RfeQ8XjOIl"
      },
      "source": [
        "Support Vector Classifier"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "r9jpyLA5M74Y",
        "outputId": "fcdd17ff-d135-484f-9100-40aa18ff7a97"
      },
      "source": [
        "model = SVC()\n",
        "model.fit(X_train_scaled, y_train)\n",
        "predictions = model.predict(X_test_scaled)\n",
        "print(classification_report(y_test, predictions))"
      ],
      "execution_count": 33,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "              precision    recall  f1-score   support\n",
            "\n",
            "           0       0.88      0.98      0.93       470\n",
            "           1       0.73      0.27      0.40        88\n",
            "\n",
            "    accuracy                           0.87       558\n",
            "   macro avg       0.80      0.63      0.66       558\n",
            "weighted avg       0.85      0.87      0.84       558\n",
            "\n"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "ARuzyOztz7ra"
      },
      "source": [
        "*classifiers with f1_score more than 0.5 are selected as potential classifiers*"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "-1IO3dwpLXtP"
      },
      "source": [
        "# Feature Engineering"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "DBJmoBafeAfe"
      },
      "source": [
        "**creating new feature**"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "xYCEoq8LHaeB"
      },
      "source": [
        "X_train_new = X_train.copy()\n",
        "X_test_new = X_test.copy()"
      ],
      "execution_count": 34,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "dEgCVPgU1BQr"
      },
      "source": [
        "# creating a new feature by summation of 6 features\n",
        "X_train_new['Total_Spent'] = X_train_new.loc[: ,\"MntWines\":\"MntGoldProds\"].sum(axis=1)\n",
        "X_test_new['Total_Spent'] = X_test_new.loc[: ,\"MntWines\":\"MntGoldProds\"].sum(axis=1)\n",
        "\n",
        "X_train_new.drop(X_train_new.loc[:, \"MntWines\":\"MntGoldProds\"], axis=1, inplace=True)\n",
        "X_test_new.drop(X_test_new.loc[:, \"MntWines\":\"MntGoldProds\"], axis=1, inplace=True)"
      ],
      "execution_count": 35,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "5bPb1atXI_cz"
      },
      "source": [
        "AdaBoost Classifier"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "GI_ZT8GCHtkj",
        "outputId": "5303ff25-e900-425b-a65d-b9b1574fa7d5"
      },
      "source": [
        "# fitting the new data to the model\n",
        "model = AdaBoostClassifier()\n",
        "model.fit(X_train_new, y_train)\n",
        "predictions = model.predict(X_test_new)\n",
        "print(classification_report(y_test, predictions))"
      ],
      "execution_count": 36,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "              precision    recall  f1-score   support\n",
            "\n",
            "           0       0.90      0.96      0.93       470\n",
            "           1       0.68      0.45      0.54        88\n",
            "\n",
            "    accuracy                           0.88       558\n",
            "   macro avg       0.79      0.71      0.74       558\n",
            "weighted avg       0.87      0.88      0.87       558\n",
            "\n"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "Ua8MYP94JDas"
      },
      "source": [
        "Gradient Boosting Classifier"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "nTLPR4Tm1QSI",
        "outputId": "80000b8f-3f76-4198-c5c9-6777e05ebf8f"
      },
      "source": [
        "model = GradientBoostingClassifier()\n",
        "model.fit(X_train_new, y_train)\n",
        "predictions = model.predict(X_test_new)\n",
        "print(classification_report(y_test, predictions))"
      ],
      "execution_count": 37,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "              precision    recall  f1-score   support\n",
            "\n",
            "           0       0.90      0.97      0.93       470\n",
            "           1       0.74      0.42      0.54        88\n",
            "\n",
            "    accuracy                           0.89       558\n",
            "   macro avg       0.82      0.70      0.74       558\n",
            "weighted avg       0.87      0.89      0.87       558\n",
            "\n"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "PK1I-xuSJScO"
      },
      "source": [
        "XGB Classifier"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "DpGrwx0lIgaJ",
        "outputId": "b6b96359-3419-4b74-e83d-75b6d0a2e9c3"
      },
      "source": [
        "model = XGBClassifier()\n",
        "model.fit(X_train_new, y_train)\n",
        "predictions = model.predict(X_test_new)\n",
        "print(classification_report(y_test, predictions))"
      ],
      "execution_count": 38,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "              precision    recall  f1-score   support\n",
            "\n",
            "           0       0.90      0.97      0.93       470\n",
            "           1       0.71      0.41      0.52        88\n",
            "\n",
            "    accuracy                           0.88       558\n",
            "   macro avg       0.80      0.69      0.72       558\n",
            "weighted avg       0.87      0.88      0.87       558\n",
            "\n"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "_9j7-NYX3m21"
      },
      "source": [
        "*GB Classifier is improved, so adding the new feature could be effective*"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "iRmXfD78LzIN"
      },
      "source": [
        "**feature selection**"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 876
        },
        "id": "wKI4bqC0NI1k",
        "outputId": "8c79c91d-b187-4e80-efa0-fe91e6266398"
      },
      "source": [
        "# calculating mutual information scores to find the most effective features \n",
        "def make_mi_scores(X, y):\n",
        "    mi_scores = mutual_info_classif(X, y, random_state=1)\n",
        "    mi_scores = pd.Series(mi_scores, name=\"MI Scores\", index=X.columns)\n",
        "    mi_scores = mi_scores.sort_values(ascending=False)\n",
        "    return mi_scores\n",
        "\n",
        "mi_scores = make_mi_scores(X_train_new, y_train)\n",
        "\n",
        "def plot_mi_scores(scores):\n",
        "    scores = scores.sort_values(ascending=True)\n",
        "    width = np.arange(len(scores))\n",
        "    ticks = list(scores.index)\n",
        "    plt.barh(width, scores)\n",
        "    plt.yticks(width, ticks)\n",
        "    plt.title(\"Mutual Information Scores\")\n",
        "\n",
        "plt.figure(dpi=100, figsize=(10, 10))\n",
        "plot_mi_scores(mi_scores)"
      ],
      "execution_count": 55,
      "outputs": [
        {
          "output_type": "display_data",
          "data": {
            "image/png": "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\n",
            "text/plain": [
              "<Figure size 1000x1000 with 1 Axes>"
            ]
          },
          "metadata": {
            "needs_background": "light"
          }
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "IiZaxt2JNJPl"
      },
      "source": [
        "# selecting the features with mi_score more than zero \n",
        "features = mi_scores[mi_scores>0].index"
      ],
      "execution_count": 40,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "IXbzca6ZOvx9"
      },
      "source": [
        "X_train_selected = X_train_new[features].copy()\n",
        "X_test_selected = X_test_new[features].copy()"
      ],
      "execution_count": 41,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "JYQy4Gu_Jecb"
      },
      "source": [
        "AdaBoost Classifier"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "OuV__O0_KIC_",
        "outputId": "5cd9bc20-7fa2-4527-edf2-f66ad4799a5c"
      },
      "source": [
        "# fitting the selected features to the model\n",
        "model = AdaBoostClassifier()\n",
        "model.fit(X_train_selected, y_train)\n",
        "predictions = model.predict(X_test_selected)\n",
        "print(classification_report(y_test, predictions))"
      ],
      "execution_count": 42,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "              precision    recall  f1-score   support\n",
            "\n",
            "           0       0.91      0.96      0.93       470\n",
            "           1       0.67      0.48      0.56        88\n",
            "\n",
            "    accuracy                           0.88       558\n",
            "   macro avg       0.79      0.72      0.74       558\n",
            "weighted avg       0.87      0.88      0.87       558\n",
            "\n"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "Cp-mzI8jJkWQ"
      },
      "source": [
        "Gradient Boosting Classifier"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "ndk90OCHOv0k",
        "outputId": "c068a7f3-926f-4e45-c946-10db240739f6"
      },
      "source": [
        "model = GradientBoostingClassifier()\n",
        "model.fit(X_train_selected, y_train)\n",
        "predictions = model.predict(X_test_selected)\n",
        "print(classification_report(y_test, predictions))"
      ],
      "execution_count": 43,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "              precision    recall  f1-score   support\n",
            "\n",
            "           0       0.90      0.97      0.94       470\n",
            "           1       0.76      0.43      0.55        88\n",
            "\n",
            "    accuracy                           0.89       558\n",
            "   macro avg       0.83      0.70      0.74       558\n",
            "weighted avg       0.88      0.89      0.88       558\n",
            "\n"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "Yojd4TuFJpNt"
      },
      "source": [
        "XGB Classifier"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "bxUL-wzAK099",
        "outputId": "d51ed5f1-3ff6-4d1e-f72d-4edb279d875d"
      },
      "source": [
        "model = XGBClassifier()\n",
        "model.fit(X_train_selected, y_train)\n",
        "predictions = model.predict(X_test_selected)\n",
        "print(classification_report(y_test, predictions))"
      ],
      "execution_count": 44,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "              precision    recall  f1-score   support\n",
            "\n",
            "           0       0.89      0.98      0.93       470\n",
            "           1       0.77      0.39      0.52        88\n",
            "\n",
            "    accuracy                           0.89       558\n",
            "   macro avg       0.83      0.68      0.73       558\n",
            "weighted avg       0.88      0.89      0.87       558\n",
            "\n"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "RCFhNyMBLf6l"
      },
      "source": [
        "GB and AdaBoost Classifiers are improved, so feature selection could be effective"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "htdANMzWrT7i"
      },
      "source": [
        "**PCA**"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "rF6mRoh9IWvf"
      },
      "source": [
        "#scaling the data\n",
        "scaler = MinMaxScaler()\n",
        "\n",
        "X_train_selected_scaled = X_train_selected.copy()\n",
        "X_test_selected_scaled = X_test_selected.copy()\n",
        "\n",
        "X_train_selected_scaled.loc[:] = scaler.fit_transform(X_train_selected)\n",
        "X_test_selected_scaled.loc[:] = scaler.transform(X_test_selected)"
      ],
      "execution_count": 45,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "C8stdPygIW2x"
      },
      "source": [
        "#fitting the scaled data to PCA\n",
        "pca = PCA(n_components=5)\n",
        "x_train_pca = pca.fit_transform(X_train_selected_scaled)\n",
        "x_test_pca = pca.transform(X_test_selected_scaled)"
      ],
      "execution_count": 46,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "S9rsvVZhIW9C",
        "outputId": "88181dc0-2281-4c9e-b654-3c21f2dda69e"
      },
      "source": [
        "pca.explained_variance_ratio_"
      ],
      "execution_count": 47,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "array([0.33989014, 0.14827436, 0.07585536, 0.06431959, 0.05890411])"
            ]
          },
          "metadata": {},
          "execution_count": 47
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "C_Cr0yI1ucHU"
      },
      "source": [
        "*These ratios do not seem promising*"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "ChNnwcCK2C8X"
      },
      "source": [
        "# Hyperparameter Tuning"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "UwPrlpz4J0Z8"
      },
      "source": [
        "AdaBoost Classifier"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "2S52F8dMMCeQ",
        "outputId": "1f8914c2-fae8-4deb-bce7-fe44bfea493c"
      },
      "source": [
        "#optimizing the classifier by tuning the learning_rate\n",
        "model = AdaBoostClassifier(learning_rate=0.985)\n",
        "model.fit(X_train_selected, y_train)\n",
        "predictions = model.predict(X_test_selected)\n",
        "print(classification_report(y_test, predictions))"
      ],
      "execution_count": 48,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "              precision    recall  f1-score   support\n",
            "\n",
            "           0       0.91      0.96      0.94       470\n",
            "           1       0.73      0.51      0.60        88\n",
            "\n",
            "    accuracy                           0.89       558\n",
            "   macro avg       0.82      0.74      0.77       558\n",
            "weighted avg       0.88      0.89      0.88       558\n",
            "\n"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "TF0tf1NgJ4C7"
      },
      "source": [
        "Gradient Boosting Classifier"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "Sm_UOYqCzMx4",
        "outputId": "3407eb0a-2420-4cec-e337-6520af5fdf82"
      },
      "source": [
        "model = GradientBoostingClassifier(learning_rate=0.45)\n",
        "model.fit(X_train_selected, y_train)\n",
        "predictions = model.predict(X_test_selected)\n",
        "\n",
        "print(model.score(X_train_selected, y_train), model.score(X_test_selected, y_test))\n",
        "print(classification_report(y_test, predictions))"
      ],
      "execution_count": 49,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "0.9898446833930705 0.9086021505376344\n",
            "              precision    recall  f1-score   support\n",
            "\n",
            "           0       0.92      0.97      0.95       470\n",
            "           1       0.78      0.58      0.67        88\n",
            "\n",
            "    accuracy                           0.91       558\n",
            "   macro avg       0.85      0.77      0.81       558\n",
            "weighted avg       0.90      0.91      0.90       558\n",
            "\n"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "_JAMnc2ZJ_67"
      },
      "source": [
        "XGB Classifier"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "KNoxrBSyOIW-",
        "outputId": "8968da09-5c47-4a24-a1af-282add232b54"
      },
      "source": [
        "model = XGBClassifier(learning_rate=0.945)\n",
        "\n",
        "model.fit(X_train_selected, y_train)\n",
        "predictions = model.predict(X_test_selected)\n",
        "\n",
        "print(model.score(X_train_selected, y_train), model.score(X_test_selected, y_test))\n",
        "print(classification_report(y_test, predictions))"
      ],
      "execution_count": 50,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "0.992831541218638 0.899641577060932\n",
            "              precision    recall  f1-score   support\n",
            "\n",
            "           0       0.93      0.96      0.94       470\n",
            "           1       0.72      0.59      0.65        88\n",
            "\n",
            "    accuracy                           0.90       558\n",
            "   macro avg       0.82      0.77      0.80       558\n",
            "weighted avg       0.89      0.90      0.90       558\n",
            "\n"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "Sw_452NFQGp1"
      },
      "source": [
        "both of the Gradient Boost and XGBoost could be the best classifiers so far"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "XlxLhZ_kQUTL"
      },
      "source": [
        "#tunning more hyperparameters\n"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "gbk2eFqeKOv7"
      },
      "source": [
        "Gradient Boosting classifier"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "61ByPukzQZ6P",
        "outputId": "fee581be-2465-4687-f50b-aaca6ff993a5"
      },
      "source": [
        "#tuning other hyperparameters to optimize the model as much as possible\n",
        "model = GradientBoostingClassifier(learning_rate=0.45,max_depth=3,\n",
        "                                   min_samples_split=6, n_estimators=100)\n",
        "model.fit(X_train_selected, y_train)\n",
        "predictions = model.predict(X_test_selected)\n",
        "\n",
        "print(classification_report(y_test, predictions))\n",
        "print('The accuracy of train data: {}'.format(model.score(X_train_selected, y_train)))\n",
        "print('The accuracy of test data: {}'.format(model.score(X_test_selected, y_test)))"
      ],
      "execution_count": 51,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "              precision    recall  f1-score   support\n",
            "\n",
            "           0       0.93      0.97      0.95       470\n",
            "           1       0.77      0.60      0.68        88\n",
            "\n",
            "    accuracy                           0.91       558\n",
            "   macro avg       0.85      0.78      0.81       558\n",
            "weighted avg       0.90      0.91      0.90       558\n",
            "\n",
            "The accuracy of train data: 0.9904420549581839\n",
            "The accuracy of test data: 0.9086021505376344\n"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "ym7_CyHeKUuH"
      },
      "source": [
        "XGB Classifier"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "Iy67h4mRQarG",
        "outputId": "5b564a8f-c1ec-44b7-e603-f7c2e903eab1"
      },
      "source": [
        "model = XGBClassifier(learning_rate=0.945, n_estimators=105)\n",
        "\n",
        "model.fit(X_train_selected, y_train)\n",
        "predictions = model.predict(X_test_selected)\n",
        "\n",
        "print(classification_report(y_test, predictions))\n",
        "print('The accuracy of train data: {}'.format(model.score(X_train_selected, y_train)))\n",
        "print('The accuracy of test data: {}'.format(model.score(X_test_selected, y_test)))"
      ],
      "execution_count": 52,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "              precision    recall  f1-score   support\n",
            "\n",
            "           0       0.93      0.96      0.94       470\n",
            "           1       0.74      0.60      0.66        88\n",
            "\n",
            "    accuracy                           0.90       558\n",
            "   macro avg       0.83      0.78      0.80       558\n",
            "weighted avg       0.90      0.90      0.90       558\n",
            "\n",
            "The accuracy of train data: 0.992831541218638\n",
            "The accuracy of test data: 0.9032258064516129\n"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "gc3M29R3lrlE"
      },
      "source": [
        "GB Classifier has the highest f1_score (0.68 for label 1 and 0.95 for label 0)"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "kYANX0Yz8Uap"
      },
      "source": [
        "# **Final Model**"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "Dud59NO39jn3"
      },
      "source": [
        "X = pd.concat([X_train_selected, X_test_selected])\n",
        "y = pd.concat([y_train, y_test])"
      ],
      "execution_count": 53,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "IcLdjwM28EJl",
        "outputId": "6d228a82-58a1-49a0-e51d-d3ed0441484e"
      },
      "source": [
        "final_model = GradientBoostingClassifier(learning_rate=0.45,max_depth=3,\n",
        "                                   min_samples_split=6, n_estimators=100)\n",
        "final_model.fit(X, y)"
      ],
      "execution_count": 54,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "GradientBoostingClassifier(learning_rate=0.45, min_samples_split=6)"
            ]
          },
          "metadata": {},
          "execution_count": 54
        }
      ]
    }
  ]
}
