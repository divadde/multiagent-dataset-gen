import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
from faker import Faker

# Configuration
NUM_ROWS = 1_000_000
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
faker = Faker()

# Constants
EMP_COUNT = 50_000
DEPT_COUNT = 100
LAPTOPS_PER_EMP = 2
TOT_LAPTOPS = EMP_COUNT * LAPTOPS_PER_EMP
LINES_PER_ORDER = 5
ORDER_COUNT = NUM_ROWS // LINES_PER_ORDER
PRODUCT_COUNT = 5_000
SUBCAT_COUNT = 200
MAINCAT_COUNT = 20
CUSTOMER_COUNT = 50_000
SALES_REP_COUNT = 1_000
SUPPLIER_COUNT = 2_000
BATCH_COUNT = 10_000
WAREHOUSE_ZIP_COUNT = 200
OFFICE_COUNT = 3_000
FLOOR_COUNT = 30

def generate_dataset(num_rows=NUM_ROWS):
    # Departments and cost centers (Cycle 2)
    dept_codes = np.array([f"D{idx+1:03d}" for idx in range(DEPT_COUNT)], dtype=object)
    dept_names = np.array([f"Department_{idx+1}" for idx in range(DEPT_COUNT)], dtype=object)
    cost_centers = np.array([f"CC{idx+1:04d}" for idx in range(DEPT_COUNT)], dtype=object)

    # Employees (Cycle 1 HR)
    employee_ids = np.array([f"EMP{idx+1:06d}" for idx in range(EMP_COUNT)], dtype=object)
    corporate_usernames = np.array([f"user{idx+1:06d}" for idx in range(EMP_COUNT)], dtype=object)
    company_emails = np.char.add(corporate_usernames, "@company.com")
    employee_dept_idx = np.random.randint(0, DEPT_COUNT, EMP_COUNT)
    employee_dept_codes = dept_codes[employee_dept_idx]
    employee_dept_names = dept_names[employee_dept_idx]
    employee_cost_centers = cost_centers[employee_dept_idx]

    # Laptops & Assets (Cycle 3)
    laptop_indices = np.arange(TOT_LAPTOPS)
    laptop_serials = np.array([f"LAP{idx+1:06d}" for idx in laptop_indices], dtype=object)
    hex_strings = np.array([f"{idx:012x}" for idx in laptop_indices])
    mac_addresses = np.array([":".join([h[i:i+2] for i in range(0, 12, 2)]) for h in hex_strings], dtype=object)
    asset_tags = np.array([f"AT{idx+1:07d}" for idx in laptop_indices], dtype=object)
    laptop_assigned_emp_idx = laptop_indices // LAPTOPS_PER_EMP
    laptop_assigned_emp_ids = employee_ids[laptop_assigned_emp_idx]

    # Badges (Cycle 4)
    badge_ids = np.array([f"BDG{idx+1:06d}" for idx in range(EMP_COUNT)], dtype=object)
    nfc_chip_codes = np.array([f"NFC{idx+1:06d}" for idx in range(EMP_COUNT)], dtype=object)

    # Offices and Floors
    floor_numbers = np.arange(1, FLOOR_COUNT + 1)
    wings = np.array(['North', 'South', 'East', 'West'])
    building_wing_from_floor = wings[(floor_numbers - 1) % len(wings)]
    office_room_numbers = np.array([f"R{idx+1:05d}" for idx in range(OFFICE_COUNT)], dtype=object)
    office_floor_numbers = (np.arange(OFFICE_COUNT) % FLOOR_COUNT) + 1

    # Parking (Composite key with day_of_week)
    days = np.array(['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'], dtype=object)
    parking_spots_pool = np.array([f"P{i:05d}" for i in range(1, EMP_COUNT + 1)], dtype=object)
    day_to_spots = {day: np.random.permutation(parking_spots_pool) for day in days}

    # Access Levels
    access_levels = np.array(['L1', 'L2', 'L3', 'Admin'], dtype=object)
    access_to_zone = {'L1': 'ZoneA', 'L2': 'ZoneB', 'L3': 'ZoneC', 'Admin': 'ZoneD'}

    # Products and Categories
    main_category_ids = np.array([f"MC{idx+1:03d}" for idx in range(MAINCAT_COUNT)], dtype=object)
    main_category_vat = np.round(np.random.uniform(0.05, 0.25, MAINCAT_COUNT), 3)
    sub_category_codes = np.array([f"SUB{idx+1:03d}" for idx in range(SUBCAT_COUNT)], dtype=object)
    subcat_main_idx = np.random.randint(0, MAINCAT_COUNT, SUBCAT_COUNT)
    product_ids = np.array([f"PRD{idx+1:05d}" for idx in range(PRODUCT_COUNT)], dtype=object)
    product_names = np.array([f"Product_{idx+1}" for idx in range(PRODUCT_COUNT)], dtype=object)
    product_subcat_idx = np.random.randint(0, SUBCAT_COUNT, PRODUCT_COUNT)
    product_subcats = sub_category_codes[product_subcat_idx]
    product_main_idx = subcat_main_idx[product_subcat_idx]
    product_maincats = main_category_ids[product_main_idx]
    product_vat_rates = main_category_vat[product_main_idx]
    product_base_prices = np.round(np.random.uniform(10, 500, PRODUCT_COUNT), 2)

    # Customers and Sales Reps
    sales_rep_ids = np.array([f"SR{idx+1:04d}" for idx in range(SALES_REP_COUNT)], dtype=object)
    customer_ids = np.array([f"CUST{idx+1:06d}" for idx in range(CUSTOMER_COUNT)], dtype=object)
    customer_salesrep_idx = np.random.randint(0, SALES_REP_COUNT, CUSTOMER_COUNT)
    customer_tiers = np.random.choice(['Bronze', 'Silver', 'Gold'], CUSTOMER_COUNT, p=[0.4, 0.35, 0.25])

    # Orders
    order_ids = np.array([f"ORD{idx+1:06d}" for idx in range(ORDER_COUNT)], dtype=object)
    order_customer_idx = np.random.randint(0, CUSTOMER_COUNT, ORDER_COUNT)
    base_order_date = pd.to_datetime(datetime(2021, 1, 1))
    order_dates = base_order_date + pd.to_timedelta(np.random.randint(0, 1460, ORDER_COUNT), unit='D')
    shipping_tracking_codes = np.array([f"TRK{idx+1:07d}" for idx in range(ORDER_COUNT)], dtype=object)

    # Warehouses (Geo chain)
    countries = np.array(['US', 'GB', 'DE', 'FR', 'JP'], dtype=object)
    currency_map = {'US': 'USD', 'GB': 'GBP', 'DE': 'EUR', 'FR': 'EUR', 'JP': 'JPY'}
    region_count = 20
    region_codes = np.array([f"REG{idx+1:03d}" for idx in range(region_count)], dtype=object)
    region_country_idx = np.random.randint(0, len(countries), region_count)
    region_countries = countries[region_country_idx]
    city_count = WAREHOUSE_ZIP_COUNT
    city_names = np.array([f"{faker.city()}_{idx+1}" for idx in range(city_count)], dtype=object)
    city_region_idx = np.array([idx % region_count for idx in range(city_count)])
    warehouse_zip_codes = np.array([f"ZIP{idx+1:05d}" for idx in range(WAREHOUSE_ZIP_COUNT)], dtype=object)
    warehouse_cities = city_names
    warehouse_regions = region_codes[city_region_idx]
    warehouse_countries = region_countries[city_region_idx]
    warehouse_currencies = np.array([currency_map[c] for c in warehouse_countries], dtype=object)

    # Suppliers (Cycle 5)
    supplier_ids = np.array([f"SUP{idx+1:06d}" for idx in range(SUPPLIER_COUNT)], dtype=object)
    supplier_emails = np.array([f"{faker.first_name().lower()}.{faker.last_name().lower()}{idx}@supplier.com"
                                for idx in range(SUPPLIER_COUNT)], dtype=object)

    # Batches
    batch_numbers = np.array([f"BATCH{idx+1:06d}" for idx in range(BATCH_COUNT)], dtype=object)
    base_manu_date = pd.to_datetime(datetime(2019, 1, 1))
    manufacture_dates = base_manu_date + pd.to_timedelta(np.random.randint(0, 1500, BATCH_COUNT), unit='D')
    expiration_dates = manufacture_dates + pd.to_timedelta(np.random.randint(180, 900, BATCH_COUNT), unit='D')

    # Row-level assignments
    order_idx_rows = np.repeat(np.arange(ORDER_COUNT), LINES_PER_ORDER)
    order_id_col = order_ids[order_idx_rows]
    order_line_number_col = np.tile(np.arange(1, LINES_PER_ORDER + 1), ORDER_COUNT)

    employee_idx_rows = np.random.randint(0, EMP_COUNT, num_rows)
    employee_id_col = employee_ids[employee_idx_rows]
    corporate_username_col = corporate_usernames[employee_idx_rows]
    company_email_col = company_emails[employee_idx_rows]
    emp_dept_idx_rows = employee_dept_idx[employee_idx_rows]
    department_code_col = dept_codes[emp_dept_idx_rows]
    department_name_col = dept_names[emp_dept_idx_rows]
    cost_center_id_col = cost_centers[emp_dept_idx_rows]

    laptop_idx_rows = employee_idx_rows * LAPTOPS_PER_EMP + np.random.randint(0, LAPTOPS_PER_EMP, num_rows)
    laptop_serial_col = laptop_serials[laptop_idx_rows]
    mac_address_col = mac_addresses[laptop_idx_rows]
    asset_tag_col = asset_tags[laptop_idx_rows]
    current_assigned_employee_id_col = laptop_assigned_emp_ids[laptop_idx_rows]

    office_idx_rows = np.random.randint(0, OFFICE_COUNT, num_rows)
    office_room_number_col = office_room_numbers[office_idx_rows]
    floor_number_col = office_floor_numbers[office_idx_rows]
    building_wing_col = building_wing_from_floor[floor_number_col - 1]

    day_of_week_col = np.random.choice(days, num_rows)
    assigned_parking_spot_col = np.empty(num_rows, dtype=object)
    for day in days:
        mask = day_of_week_col == day
        if mask.any():
            assigned_parking_spot_col[mask] = day_to_spots[day][employee_idx_rows[mask]]

    badge_id_col = badge_ids[employee_idx_rows]
    nfc_chip_code_col = nfc_chip_codes[employee_idx_rows]
    badge_status_col = np.random.choice(['Active', 'Inactive', 'Suspended'], num_rows, p=[0.75, 0.15, 0.10])
    access_level_col = np.random.choice(access_levels, num_rows, p=[0.4, 0.3, 0.2, 0.1])
    security_zone_col = np.vectorize(access_to_zone.get)(access_level_col)

    product_idx_rows = np.random.randint(0, PRODUCT_COUNT, num_rows)
    product_id_col = product_ids[product_idx_rows]
    product_name_col = product_names[product_idx_rows]
    sub_category_code_col = product_subcats[product_idx_rows]
    main_category_id_col = product_maincats[product_idx_rows]
    vat_tax_rate_col = product_vat_rates[product_idx_rows]

    customer_idx_rows = order_customer_idx[order_idx_rows]
    customer_id_col = customer_ids[customer_idx_rows]
    assigned_sales_rep_id_col = sales_rep_ids[customer_salesrep_idx[customer_idx_rows]]
    customer_tier_col = customer_tiers[customer_idx_rows]
    factor_map = {'Bronze': 1.0, 'Silver': 0.95, 'Gold': 0.90}
    factor_array = np.vectorize(factor_map.get)(customer_tier_col)
    unit_price_applied_col = np.round(product_base_prices[product_idx_rows] * factor_array, 2)

    order_date_col = order_dates[order_idx_rows]
    shipping_tracking_code_col = shipping_tracking_codes[order_idx_rows]

    warehouse_idx_rows = np.random.randint(0, WAREHOUSE_ZIP_COUNT, num_rows)
    warehouse_zip_code_col = warehouse_zip_codes[warehouse_idx_rows]
    warehouse_city_col = warehouse_cities[warehouse_idx_rows]
    warehouse_region_state_col = warehouse_regions[warehouse_idx_rows]
    country_iso_code_col = warehouse_countries[warehouse_idx_rows]
    local_currency_col = warehouse_currencies[warehouse_idx_rows]

    supplier_idx_rows = np.random.randint(0, SUPPLIER_COUNT, num_rows)
    supplier_id_col = supplier_ids[supplier_idx_rows]
    main_contact_email_col = supplier_emails[supplier_idx_rows]

    batch_idx_rows = np.random.randint(0, BATCH_COUNT, num_rows)
    batch_lot_number_col = batch_numbers[batch_idx_rows]
    manufacture_date_col = manufacture_dates[batch_idx_rows]
    expiration_date_col = expiration_dates[batch_idx_rows]

    df = pd.DataFrame({
        'employee_id': employee_id_col,
        'corporate_username': corporate_username_col,
        'company_email': company_email_col,
        'department_code': department_code_col,
        'department_name': department_name_col,
        'cost_center_id': cost_center_id_col,
        'laptop_serial_number': laptop_serial_col,
        'mac_address': mac_address_col,
        'asset_tag_id': asset_tag_col,
        'current_assigned_employee_id': current_assigned_employee_id_col,
        'floor_number': floor_number_col,
        'building_wing': building_wing_col,
        'office_room_number': office_room_number_col,
        'day_of_week': day_of_week_col,
        'assigned_parking_spot': assigned_parking_spot_col,
        'badge_status': badge_status_col,
        'access_level': access_level_col,
        'security_zone': security_zone_col,
        'badge_id': badge_id_col,
        'nfc_chip_code': nfc_chip_code_col,
        'order_id': order_id_col,
        'order_line_number': order_line_number_col,
        'product_id': product_id_col,
        'product_name': product_name_col,
        'sub_category_code': sub_category_code_col,
        'main_category_id': main_category_id_col,
        'vat_tax_rate': vat_tax_rate_col,
        'customer_tier': customer_tier_col,
        'unit_price_applied': unit_price_applied_col,
        'customer_id': customer_id_col,
        'assigned_sales_rep_id': assigned_sales_rep_id_col,
        'order_date': order_date_col,
        'shipping_tracking_code': shipping_tracking_code_col,
        'batch_lot_number': batch_lot_number_col,
        'manufacture_date': manufacture_date_col,
        'expiration_date': expiration_date_col,
        'warehouse_zip_code': warehouse_zip_code_col,
        'warehouse_city': warehouse_city_col,
        'warehouse_region_state': warehouse_region_state_col,
        'country_iso_code': country_iso_code_col,
        'local_currency': local_currency_col,
        'supplier_id': supplier_id_col,
        'main_contact_email': main_contact_email_col
    })

    return df


def validate_data(df: pd.DataFrame):
    print("--- VALIDATION START ---")
    assert len(df) == NUM_ROWS, "Row count mismatch"

    def fd_check(determinants, dependent, condition=None, msg=""):
        subset = df if condition is None else df[condition]
        grouped = subset.groupby(determinants)[dependent].nunique()
        assert (grouped == 1).all(), f"FD Violation: {msg}"

    fd_check(['employee_id'], 'corporate_username', msg="employee_id -> corporate_username")
    fd_check(['corporate_username'], 'company_email', msg="corporate_username -> company_email")
    fd_check(['company_email'], 'employee_id', msg="company_email -> employee_id")

    fd_check(['department_code'], 'department_name', msg="department_code -> department_name")
    fd_check(['department_name'], 'cost_center_id', msg="department_name -> cost_center_id")
    fd_check(['cost_center_id'], 'department_code', msg="cost_center_id -> department_code")
    fd_check(['employee_id'], 'department_code', msg="employee_id -> department_code")

    fd_check(['laptop_serial_number'], 'mac_address', msg="laptop_serial_number -> mac_address")
    fd_check(['mac_address'], 'asset_tag_id', msg="mac_address -> asset_tag_id")
    fd_check(['asset_tag_id'], 'laptop_serial_number', msg="asset_tag_id -> laptop_serial_number")
    fd_check(['laptop_serial_number'], 'current_assigned_employee_id', msg="laptop_serial_number -> current_assigned_employee_id")

    fd_check(['floor_number'], 'building_wing', msg="floor_number -> building_wing")
    fd_check(['office_room_number'], 'floor_number', msg="office_room_number -> floor_number")

    fd_check(['employee_id', 'day_of_week'], 'assigned_parking_spot', msg="(employee_id, day) -> parking")
    fd_check(['assigned_parking_spot', 'day_of_week'], 'employee_id', msg="(parking, day) -> employee")

    fd_check(['badge_id'], 'nfc_chip_code', msg="badge_id -> nfc_chip_code")
    fd_check(['nfc_chip_code'], 'badge_id', msg="nfc_chip_code -> badge_id")
    fd_check(['badge_id'], 'employee_id', msg="badge_id -> employee_id")
    fd_check(['access_level'], 'security_zone', condition=(df['badge_status'] == 'Active'),
             msg="access_level -> security_zone (Active)")

    fd_check(['product_id'], 'product_name', msg="product_id -> product_name")
    fd_check(['product_name'], 'sub_category_code', msg="product_name -> sub_category_code")
    fd_check(['sub_category_code'], 'main_category_id', msg="sub_category_code -> main_category_id")
    fd_check(['main_category_id'], 'vat_tax_rate', msg="main_category_id -> vat_tax_rate")

    fd_check(['warehouse_zip_code'], 'warehouse_city', msg="warehouse_zip_code -> warehouse_city")
    fd_check(['warehouse_city'], 'warehouse_region_state', msg="warehouse_city -> warehouse_region_state")
    fd_check(['warehouse_region_state'], 'country_iso_code', msg="warehouse_region_state -> country_iso_code")
    fd_check(['country_iso_code'], 'local_currency', msg="country_iso_code -> local_currency")

    fd_check(['supplier_id'], 'main_contact_email', msg="supplier_id -> main_contact_email")
    fd_check(['main_contact_email'], 'supplier_id', msg="main_contact_email -> supplier_id")

    fd_check(['order_id', 'order_line_number'], 'product_id', msg="(order_id, line) -> product_id")
    fd_check(['product_id', 'customer_tier'], 'unit_price_applied', msg="(product_id, tier) -> unit_price_applied")
    fd_check(['order_id'], 'customer_id', msg="order_id -> customer_id")
    fd_check(['order_id'], 'order_date', msg="order_id -> order_date")
    fd_check(['customer_id'], 'assigned_sales_rep_id', msg="customer_id -> assigned_sales_rep_id")
    fd_check(['shipping_tracking_code'], 'order_id', msg="shipping_tracking_code -> order_id")
    fd_check(['batch_lot_number'], 'manufacture_date', msg="batch_lot_number -> manufacture_date")
    fd_check(['batch_lot_number'], 'expiration_date', msg="batch_lot_number -> expiration_date")

    non_unique_cols = [
        'employee_id', 'laptop_serial_number', 'department_code', 'badge_id',
        'assigned_parking_spot', 'cost_center_id', 'order_id', 'product_id',
        'customer_id', 'warehouse_zip_code', 'batch_lot_number',
        'assigned_sales_rep_id', 'main_category_id'
    ]
    for col in non_unique_cols:
        assert not df[col].is_unique, f"Cardinality violation: {col} is unique"

    assert not df.duplicated(subset=['order_id', 'order_line_number']).any(), "Duplicate order_id + order_line_number"

    print("SUCCESS: All validations passed.")


df = generate_dataset(NUM_ROWS)
validate_data(df)
df.to_csv("synthetic_dataset.csv", index=False)