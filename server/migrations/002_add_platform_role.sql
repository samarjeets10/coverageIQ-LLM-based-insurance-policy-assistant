
ALTER TABLE users
ADD COLUMN platform_role VARCHAR(20) NOT NULL DEFAULT 'user'
CHECK (platform_role IN ('user', 'admin'));
