CREATE TABLE twitter.streamed_vaccine
(
    tweet_text text COLLATE pg_catalog."default",
    tweet_id text COLLATE pg_catalog."default",
    created_at timestamp without time zone,
    place_full_name text COLLATE pg_catalog."default",
    place_country_code text COLLATE pg_catalog."default",
    known_language text COLLATE pg_catalog."default",
    location_user text COLLATE pg_catalog."default"
)
WITH (
    OIDS = FALSE
)
TABLESPACE pg_default;

ALTER TABLE twitter.streamed_vaccine
    OWNER to postgres;

CREATE TABLE twitter.streamed_full
(
    tweet_text text COLLATE pg_catalog."default",
    tweet_id text COLLATE pg_catalog."default",
    created_at timestamp without time zone,
    place_full_name text COLLATE pg_catalog."default",
    place_country_code text COLLATE pg_catalog."default",
    known_language text COLLATE pg_catalog."default",
    location_user text COLLATE pg_catalog."default"
)
WITH (
    OIDS = FALSE
)
TABLESPACE pg_default;

ALTER TABLE twitter.streamed_full
    OWNER to postgres;