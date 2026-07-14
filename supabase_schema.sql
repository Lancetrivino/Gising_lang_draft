-- GisingLang: drowsiness_events table
-- Matches the CloudLogger payload exactly (device_id, event_timestamp,
-- alert_level, ear_value, perclos_value, pitch, yaw, roll), per thesis
-- Chapter 3's Network Layer description of the CloudLogger payload.
--
-- Run this in the Supabase project's SQL editor (Database > SQL Editor).

create table if not exists drowsiness_events (
    id uuid primary key default gen_random_uuid(),
    device_id text not null,
    event_timestamp timestamptz not null,
    alert_level text not null check (alert_level in ('NONE', 'LAYER1', 'LAYER2', 'COMBINED')),
    ear_value numeric not null,
    perclos_value numeric not null,
    pitch numeric not null,
    yaw numeric not null,
    roll numeric not null,
    created_at timestamptz not null default now()
);

-- Row Level Security: enabled so dashboards (using the anon/authenticated
-- key) cannot read or write this table directly. The Raspberry Pi writes
-- using the project's service_role key instead, which bypasses RLS by
-- design - keep that key only on the device (environment variable), never
-- in the dashboard code.
alter table drowsiness_events enable row level security;

-- Speeds up the dashboard's "my device's recent events" queries once the
-- Users/Devices tables and their RLS policies are added later.
create index if not exists idx_drowsiness_events_device_id
    on drowsiness_events (device_id, event_timestamp desc);
