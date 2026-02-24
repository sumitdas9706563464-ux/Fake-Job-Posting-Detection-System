import pandas as pd
import numpy as np
from faker import Faker
import random

fake = Faker()

def generate_job_postings(num_records=2000, fraud_ratio=0.1):
    data = []

    # Real job keywords and templates
    real_titles = [
        "Software Engineer", "Data Scientist", "Project Manager", "Marketing Specialist",
        "Accountant", "HR Manager", "Sales Representative", "Customer Support",
        "DevOps Engineer", "Product Designer", "UX Researcher", "Financial Analyst"
    ]

    real_companies = [
        "TechNova Solutions", "GreenEnergy Corp", "Global Finance Group", "Creative Minds Agency",
        "HealthFirst Systems", "EduTech Partners", "Swift Logistics", "Urban Retailers"
    ]

    # Fake job keywords and templates
    fake_titles = [
        "Urgent Hiring: Data Entry", "Work from Home Opportunity", "Earn Easy Money Online",
        "Administrative Assistant - Immediate Start", "Part-time Weekly Pay", "Package Handler",
        "Home-based Virtual Assistant", "Customer Service (Remote)"
    ]

    fake_descriptions = [
        "Earn up to $5000 per month working from home. No experience required. Just need a computer and internet.",
        "We are looking for immediate starters. High pay, flexible hours. All you need to do is process payments.",
        "URGENT: Reputable company seeking remote workers. Great benefits, no interview needed. Apply now!",
        "Work in your spare time. This is a legitimate opportunity to make extra cash. Contact us via WhatsApp."
    ]

    for i in range(num_records):
        is_fraudulent = 1 if random.random() < fraud_ratio else 0

        if is_fraudulent:
            title = random.choice(fake_titles)
            company = fake.company() + " Inc."
            # Add more variability to length
            desc_len = random.randint(100, 600)
            description = random.choice(fake_descriptions) + " " + fake.text(max_nb_chars=desc_len)
            location = fake.city() + ", " + fake.country_code()
            requirements = "Must have a bank account and be able to work 10 hours a week. " + fake.sentence()
        else:
            title = random.choice(real_titles) + " " + random.choice(["II", "Senior", "Lead", ""])
            company = random.choice(real_companies)
            desc_len = random.randint(100, 600)
            description = (
                f"We are looking for a {title} to join our team at {company}. "
                "The ideal candidate will have strong communication skills and experience in the field. "
                "You will be responsible for managing projects and collaborating with cross-functional teams. "
                + fake.text(max_nb_chars=desc_len)
            )
            location = fake.city() + ", " + fake.country_code()
            requirements = "Bachelor's degree in a related field and 3+ years of experience. Proficiency in relevant tools. " + fake.sentence()

        data.append({
            "job_id": i + 1,
            "title": title.strip(),
            "location": location,
            "department": fake.job() if random.random() > 0.5 else np.nan,
            "salary_range": f"{random.randint(30, 60)}k-{random.randint(70, 120)}k" if random.random() > 0.7 else np.nan,
            "company_profile": fake.paragraph() if not is_fraudulent else np.nan,
            "description": description,
            "requirements": requirements,
            "benefits": fake.sentence() if random.random() > 0.5 else np.nan,
            "telecommuting": random.randint(0, 1),
            "has_company_logo": random.randint(0, 1) if not is_fraudulent else 0,
            "has_questions": random.randint(0, 1),
            "employment_type": random.choice(["Full-time", "Part-time", "Contract", "Other"]),
            "required_experience": random.choice(["Entry level", "Mid-Senior level", "Director", "Executive"]),
            "required_education": random.choice(["Bachelor's Degree", "Master's Degree", "High School or equivalent"]),
            "industry": fake.bs(),
            "function": fake.job(),
            "fraudulent": is_fraudulent
        })

    df = pd.DataFrame(data)
    return df

if __name__ == "__main__":
    df = generate_job_postings(num_records=5000, fraud_ratio=0.15)
    df.to_csv("data/fake_job_postings.csv", index=False)
    print(f"Generated {len(df)} records in data/fake_job_postings.csv")
    print(f"Fraudulent count: {df['fraudulent'].sum()}")
