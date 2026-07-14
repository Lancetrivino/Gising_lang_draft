-- GisingLang: users + emergency_contacts tables
-- Run this in the Supabase SQL Editor, in addition to supabase_schema.sql
-- (drowsiness_events), which you've already run.
--
-- Matches thesis Chapter 3's Database Design section: EmergencyContacts has
-- ContactID (PK), UserID (FK -> Users), Name, PhoneNumber, Relationship,
-- CreatedAt, IsActive.

-- Minimal Users table for now - just enough for the EmergencyContacts
-- foreign key to work. Once the dashboard login flow is built, this will
-- likely be extended to reference Supabase Auth's auth.users(id) directly.
create table if not exists users (
    user_id uuid primary key default gen_random_uuid(),
    role text not null check (role in ('Driver', 'Administrator')),
    created_at timestamptz not null default now()
);

create table if not exists emergency_contacts (
    contact_id uuid primary key default gen_random_uuid(),
    user_id uuid not null references users(user_id) on delete cascade,
    name varchar(255) not null,
    phone_number varchar(20) not null,
    relationship varchar(100),
    created_at timestamptz not null default now(),
    is_active boolean not null default true
);

-- RLS enabled per thesis: "drivers can only view and manage their own
-- contacts, and no other user including system administrators can access
-- or query this table." The device itself reads via the service_role key,
-- which bypasses RLS by design (same pattern as CloudLogger's writes) -
-- these policies protect the *dashboard's* access path once driver login
-- is implemented.
alter table users enable row level security;
alter table emergency_contacts enable row level security;

create index if not exists idx_emergency_contacts_user_id
    on emergency_contacts (user_id)
    where is_active = true;

-- Two sample rows so you can test CloudNotifier against real data.
-- Replace the phone number with one you've verified in your Twilio trial
-- account (Phone Numbers > Verified Caller IDs) before sending real SMS.
insert into users (user_id, role) values
    ('11111111-1111-1111-1111-111111111111', 'Driver')
on conflict (user_id) do nothing;

insert into emergency_contacts (user_id, name, phone_number, relationship, is_active) values
    ('11111111-1111-1111-1111-111111111111', 'Test Contact', '+639170000000', 'Parent', true)
on conflict do nothing;
