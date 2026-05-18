import os
import pandas as pd


def read_csv_safe(path):
    try:
        return pd.read_csv(path, encoding="utf-8-sig")
    except UnicodeDecodeError:
        print(f"utf-8-sig 读取失败，改用 ignore: {path}")
        return pd.read_csv(path, encoding="utf-8-sig", encoding_errors="ignore")

def main():
    file1 = "data/processed/csic2012_train.csv"
    file2 = "data/processed/ecml_pkdd_all.csv"
    output_path = "data/processed/merged_two.csv"

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    df1 = read_csv_safe(file1)
    df2 = read_csv_safe(file2)

    df = pd.concat([df1, df2], ignore_index=True)

    label_map = {
        # Normal
        "normal": "Normal",
        "Normal": "Normal",
        "Valid": "Normal",
        "valid": "Normal",

        # SQLi
        "SqlInjection": "SQLi",
        "SQLi": "SQLi",
        "SQL Injection": "SQLi",
        "sql injection": "SQLi",

        # XSS
        "XSS": "XSS",
        "xss": "XSS",

        # SSI
        "SSI": "SSI",
        "ssi": "SSI",

        # XPath
        "XpathInjection": "XPath",
        "XPathInjection": "XPath",
        "XPath": "XPath",

        # LDAPi
        "LdapInjection": "LDAPi",
        "LDAPi": "LDAPi",
        "LDAPInjection": "LDAPi",

        # Path Traversal
        "PathTransversal": "PathTraversal",
        "Path Traversal": "PathTraversal",
        "PathTraversal": "PathTraversal",

        # OS Command Injection
        "OsCommanding": "OSCommandInjection",
        "OS Command Injection": "OSCommandInjection",
        "OSCommandInjection": "OSCommandInjection",
        "Command Injection": "OSCommandInjection",
    }

    df["raw_label"] = df["label"].astype(str).str.strip()
    df["label"] = df["raw_label"].map(label_map)

    unknown = df[df["label"].isna()]["raw_label"].value_counts()
    if len(unknown) > 0:
        print("未映射标签：")
        print(unknown)

    keep_labels = {
        "Normal",
        "SQLi",
        "XSS",
        "SSI",
        "XPath",
        "LDAPi",
        "PathTraversal",
        "OSCommandInjection",
    }

    df = df[df["label"].isin(keep_labels)].copy()
    df = df.drop(columns=["raw_label"])
    df = df.drop_duplicates().reset_index(drop=True)

    df.to_csv(output_path, index=False, encoding="utf-8-sig")

    print("合并完成")
    print(f"输出文件: {output_path}")
    print(df["label"].value_counts())

if __name__ == "__main__":
    main()